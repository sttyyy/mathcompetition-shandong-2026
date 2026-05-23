"""
Problem 2 solver: multi-objective attraction selection and 5-day baseline itinerary.

Dependencies:
    pip install numpy pandas matplotlib openpyxl

This script uses the processed data and Problem 1 outputs already generated in
data/processed. It does not repeat Problem 1 data cleaning or modeling.

Solving logic:
    1. Data input: load attractions, hotel/attraction commute matrices and TOPSIS scores.
    2. Parameter initialization: set daily time limits, meal duration, population size, etc.
    3. Model call: generate feasible route individuals and apply non-dominated sorting.
    4. Result output: export Pareto alternatives, final baseline itinerary and visual charts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"


@dataclass(frozen=True)
class Problem2Config:
    """Centralized parameters; tune them here instead of scattering constants."""

    days: int = 5
    min_spots: int = 5
    max_spots: int = 8
    day_start: float = 7.0
    day_end: float = 21.0
    prep_hours: float = 1.5
    meal_hours: float = 1.0
    max_pair_commute_min: float = 60.0
    population_size: int = 80
    generations: int = 120
    crossover_rate: float = 0.9
    mutation_rate: float = 0.15
    seed: int = 20260523
    final_weights: tuple[float, float, float] = (0.45, 0.30, 0.25)


def ensure_dirs() -> None:
    """Create output folders if they do not exist."""

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """Configure Chinese display for matplotlib charts."""

    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def time_to_text(value: float) -> str:
    """Convert decimal hour to HH:MM text; this keeps output tables readable."""

    minutes = int(round(value * 60))
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def minmax(values: Iterable[float], larger_is_better: bool = True) -> np.ndarray:
    """Min-max normalize values into [0, 1].

    np.asarray is used because it keeps vector operations stable and concise.
    """

    arr = np.asarray(list(values), dtype=float)
    span = arr.max() - arr.min()
    if math.isclose(float(span), 0.0):
        return np.ones_like(arr)
    scaled = (arr - arr.min()) / span
    return scaled if larger_is_better else 1.0 - scaled


def load_problem_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 1: data input from processed Problem 1 files."""

    required = [
        PROCESSED_DIR / "attractions_processed.csv",
        PROCESSED_DIR / "hotel_commute_minutes.csv",
        PROCESSED_DIR / "attraction_commute_minutes.csv",
        PROCESSED_DIR / "problem1_latest_topsis_result.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required processed files:\n" + "\n".join(missing))

    attractions = pd.read_csv(required[0], encoding="utf-8-sig")
    hotel_commute = pd.read_csv(required[1], index_col=0, encoding="utf-8-sig")
    commute = pd.read_csv(required[2], index_col=0, encoding="utf-8-sig")
    topsis = pd.read_csv(required[3], encoding="utf-8-sig")

    attractions = attractions.merge(
        topsis[["景点ID", "TOPSIS贴近度", "优先级"]],
        left_on="id",
        right_on="景点ID",
        how="left",
    )
    attractions = attractions.drop(columns=["景点ID"])
    return attractions, hotel_commute, commute, topsis


def simulate_day(
    route: tuple[str, ...],
    attractions: pd.DataFrame,
    hotel_commute: pd.DataFrame,
    commute: pd.DataFrame,
    cfg: Problem2Config,
) -> dict | None:
    """Build one day's timeline and return None if opening/time constraints fail."""

    spot = attractions.set_index("id")
    current_time = cfg.day_start
    events: list[dict] = []

    prep_start = current_time
    current_time += cfg.prep_hours
    events.append({"环节": "起床早餐与整装", "开始": prep_start, "结束": current_time, "地点": "酒店"})

    last_place = "酒店"
    driving_minutes = 0.0
    visited_names: list[str] = []

    for index, spot_id in enumerate(route):
        if last_place == "酒店":
            travel_min = float(hotel_commute.loc["酒店", spot_id])
        else:
            travel_min = float(commute.loc[last_place, spot_id])

        travel_start = current_time
        current_time += travel_min / 60.0
        driving_minutes += travel_min
        events.append(
            {
                "环节": "酒店至景点通勤" if last_place == "酒店" else "景点间通勤",
                "开始": travel_start,
                "结束": current_time,
                "地点": f"{last_place}->{spot_id}",
            }
        )

        open_start = float(spot.loc[spot_id, "effective_open_start"])
        open_end = float(spot.loc[spot_id, "effective_open_end"])
        if current_time < open_start:
            wait_start = current_time
            current_time = open_start
            events.append({"环节": "等待开园", "开始": wait_start, "结束": current_time, "地点": spot_id})

        visit_start = current_time
        current_time += float(spot.loc[spot_id, "comfort_time"])
        visit_end = current_time
        if visit_start < open_start or visit_end > open_end:
            return None

        visited_names.append(str(spot.loc[spot_id, "name"]))
        events.append({"环节": "景点游览", "开始": visit_start, "结束": visit_end, "地点": spot_id})

        # For two-spot days, lunch is placed between attractions to avoid cutting a visit.
        if index == 0 and len(route) == 2:
            meal_start = current_time
            current_time += cfg.meal_hours
            events.append({"环节": "午餐", "开始": meal_start, "结束": current_time, "地点": "途中/景区周边"})

        last_place = spot_id

    return_min = float(hotel_commute.loc["酒店", last_place])
    return_start = current_time
    current_time += return_min / 60.0
    driving_minutes += return_min
    events.append({"环节": "返程至酒店", "开始": return_start, "结束": current_time, "地点": f"{last_place}->酒店"})

    # Single-spot days still reserve one formal meal if the day would otherwise end too early.
    if len(route) == 1 and current_time <= cfg.day_end - cfg.meal_hours:
        meal_start = current_time
        current_time += cfg.meal_hours
        events.append({"环节": "正餐", "开始": meal_start, "结束": current_time, "地点": "酒店/周边"})

    if current_time > cfg.day_end or current_time - cfg.day_start > (cfg.day_end - cfg.day_start):
        return None

    return {
        "route": route,
        "events": events,
        "spot_count": len(route),
        "spot_names": "、".join(visited_names),
        "driving_minutes": driving_minutes,
        "duration_hours": current_time - cfg.day_start,
        "end_time": current_time,
    }


def build_day_options(
    attractions: pd.DataFrame,
    hotel_commute: pd.DataFrame,
    commute: pd.DataFrame,
    cfg: Problem2Config,
) -> list[dict]:
    """Generate all feasible one-day single/pair routes.

    Pair routes are capped by max_pair_commute_min because the model requires
    same-day combinations to be strong or weak linkage pairs from Problem 1.
    """

    ids = attractions["id"].tolist()
    options: list[dict] = []

    for spot_id in ids:
        result = simulate_day((spot_id,), attractions, hotel_commute, commute, cfg)
        if result is not None:
            options.append(result)

    for i in ids:
        for j in ids:
            if i == j:
                continue
            if float(commute.loc[i, j]) > cfg.max_pair_commute_min:
                continue
            result = simulate_day((i, j), attractions, hotel_commute, commute, cfg)
            if result is not None:
                options.append(result)

    return options


def repair_individual(
    individual: list[dict],
    day_options: list[dict],
    cfg: Problem2Config,
    rng: random.Random,
) -> list[dict]:
    """Repair duplicated spots and total-spot violations after genetic operations."""

    repaired: list[dict] = []
    used: set[str] = set()

    for route in individual:
        ids = set(route["route"])
        if ids & used:
            feasible = [opt for opt in day_options if not (set(opt["route"]) & used)]
            route = rng.choice(feasible) if feasible else route
            ids = set(route["route"])
        repaired.append(route)
        used |= ids

    def total_spots(routes: list[dict]) -> int:
        return len({sid for opt in routes for sid in opt["route"]})

    # If too many spots were selected, replace pair-days with single-days.
    while total_spots(repaired) > cfg.max_spots:
        pair_indices = [idx for idx, opt in enumerate(repaired) if len(opt["route"]) == 2]
        if not pair_indices:
            break
        idx = rng.choice(pair_indices)
        used_without_day = {sid for k, opt in enumerate(repaired) if k != idx for sid in opt["route"]}
        candidates = [opt for opt in day_options if len(opt["route"]) == 1 and not (set(opt["route"]) & used_without_day)]
        if not candidates:
            break
        repaired[idx] = rng.choice(candidates)

    # If too few spots were selected, replace single-days with pair-days where possible.
    while total_spots(repaired) < cfg.min_spots:
        single_indices = [idx for idx, opt in enumerate(repaired) if len(opt["route"]) == 1]
        if not single_indices:
            break
        idx = rng.choice(single_indices)
        used_without_day = {sid for k, opt in enumerate(repaired) if k != idx for sid in opt["route"]}
        candidates = [opt for opt in day_options if len(opt["route"]) == 2 and not (set(opt["route"]) & used_without_day)]
        if not candidates:
            break
        repaired[idx] = rng.choice(candidates)

    return repaired


def random_individual(day_options: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """Create one feasible 5-day itinerary."""

    for _ in range(2000):
        routes = rng.sample(day_options, cfg.days)
        repaired = repair_individual(routes, day_options, cfg, rng)
        selected = {sid for opt in repaired for sid in opt["route"]}
        if cfg.min_spots <= len(selected) <= cfg.max_spots and len(selected) == sum(len(opt["route"]) for opt in repaired):
            return repaired
    raise RuntimeError("Could not initialize a feasible itinerary; check constraints.")


def evaluate(individual: list[dict], attractions: pd.DataFrame) -> dict:
    """Calculate objectives F1, F2, F3 and helper metrics for one itinerary."""

    spot = attractions.set_index("id")
    selected = [sid for opt in individual for sid in opt["route"]]
    daily_hours = np.array([opt["duration_hours"] for opt in individual], dtype=float)
    daily_drive = np.array([opt["driving_minutes"] for opt in individual], dtype=float)

    # np.mean avoids manual sum/n mistakes and is clearer for vector data.
    total_preference = float(spot.loc[selected, "preference"].sum())
    total_topsis = float(spot.loc[selected, "TOPSIS贴近度"].sum())
    total_drive = float(daily_drive.sum())
    balance_var = float(np.mean((daily_hours - np.mean(daily_hours)) ** 2))

    return {
        "selected_ids": selected,
        "selected_count": len(selected),
        "total_preference": total_preference,
        "total_topsis": total_topsis,
        "total_drive_min": total_drive,
        "balance_variance": balance_var,
        "daily_hours": daily_hours,
        "daily_drive": daily_drive,
    }


def dominates(a: dict, b: dict) -> bool:
    """Return True if solution a Pareto-dominates solution b."""

    better_or_equal = (
        a["total_preference"] >= b["total_preference"]
        and a["total_drive_min"] <= b["total_drive_min"]
        and a["balance_variance"] <= b["balance_variance"]
    )
    strictly_better = (
        a["total_preference"] > b["total_preference"]
        or a["total_drive_min"] < b["total_drive_min"]
        or a["balance_variance"] < b["balance_variance"]
    )
    return better_or_equal and strictly_better


def non_dominated_front(population: list[dict]) -> list[dict]:
    """Extract the first Pareto front."""

    front = []
    for candidate in population:
        if not any(dominates(other["metrics"], candidate["metrics"]) for other in population if other is not candidate):
            front.append(candidate)
    return front


def crossover(parent1: list[dict], parent2: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """One-point crossover on the 5-day route list."""

    point = rng.randint(1, cfg.days - 1)
    return parent1[:point] + parent2[point:]


def mutate(individual: list[dict], day_options: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """Randomly replace one day's route; repair keeps the solution feasible."""

    mutated = list(individual)
    idx = rng.randrange(cfg.days)
    mutated[idx] = rng.choice(day_options)
    return repair_individual(mutated, day_options, cfg, rng)


def score_for_final_choice(pareto: pd.DataFrame, cfg: Problem2Config) -> np.ndarray:
    """Weighted score used only to select one representative baseline from Pareto set."""

    w1, w2, w3 = cfg.final_weights
    satisfaction = minmax(pareto["总喜好度"], larger_is_better=True)
    drive = minmax(pareto["总行车时间min"], larger_is_better=False)
    balance = minmax(pareto["日耗时方差"], larger_is_better=False)
    return w1 * satisfaction + w2 * drive + w3 * balance


def solve_problem2(
    attractions: pd.DataFrame,
    hotel_commute: pd.DataFrame,
    commute: pd.DataFrame,
    cfg: Problem2Config,
) -> tuple[list[dict], dict, list[dict]]:
    """Step 3: model call using genetic search and Pareto filtering."""

    rng = random.Random(cfg.seed)
    day_options = build_day_options(attractions, hotel_commute, commute, cfg)
    population = [
        {"routes": random_individual(day_options, cfg, rng)}
        for _ in range(cfg.population_size)
    ]

    for item in population:
        item["metrics"] = evaluate(item["routes"], attractions)

    history = []
    for generation in range(1, cfg.generations + 1):
        front = non_dominated_front(population)
        best_pref = max(item["metrics"]["total_preference"] for item in front)
        best_drive = min(item["metrics"]["total_drive_min"] for item in front)
        best_balance = min(item["metrics"]["balance_variance"] for item in front)
        history.append(
            {
                "迭代代数": generation,
                "Pareto解数量": len(front),
                "当前最高喜好度": best_pref,
                "当前最低行车时间min": best_drive,
                "当前最低日耗时方差": best_balance,
            }
        )

        next_population = list(front)
        while len(next_population) < cfg.population_size:
            parent1, parent2 = rng.sample(population, 2)
            if rng.random() < cfg.crossover_rate:
                child_routes = crossover(parent1["routes"], parent2["routes"], cfg, rng)
            else:
                child_routes = list(parent1["routes"])
            if rng.random() < cfg.mutation_rate:
                child_routes = mutate(child_routes, day_options, cfg, rng)
            child_routes = repair_individual(child_routes, day_options, cfg, rng)
            selected = [sid for opt in child_routes for sid in opt["route"]]
            if cfg.min_spots <= len(set(selected)) <= cfg.max_spots and len(set(selected)) == len(selected):
                next_population.append({"routes": child_routes, "metrics": evaluate(child_routes, attractions)})

        population = next_population[: cfg.population_size]

    pareto = non_dominated_front(population)
    return pareto, {"history": history, "day_options": day_options}, population


def build_output_tables(
    pareto: list[dict],
    attractions: pd.DataFrame,
    cfg: Problem2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Step 4: transform model results into report-ready tables."""

    spot = attractions.set_index("id")
    rows = []
    for idx, item in enumerate(pareto, start=1):
        metrics = item["metrics"]
        rows.append(
            {
                "方案编号": f"P{idx}",
                "优选景点数": metrics["selected_count"],
                "优选景点": "、".join(metrics["selected_ids"]),
                "优选景点名称": "、".join(spot.loc[metrics["selected_ids"], "name"].astype(str).tolist()),
                "总喜好度": metrics["total_preference"],
                "TOPSIS总贴近度": metrics["total_topsis"],
                "总行车时间min": metrics["total_drive_min"],
                "日均行车时间min": metrics["total_drive_min"] / cfg.days,
                "日耗时方差": metrics["balance_variance"],
            }
        )
    pareto_df = pd.DataFrame(rows).drop_duplicates(subset=["优选景点", "总行车时间min", "日耗时方差"])
    pareto_df["综合筛选得分"] = score_for_final_choice(pareto_df, cfg)
    pareto_df = pareto_df.sort_values("综合筛选得分", ascending=False).reset_index(drop=True)
    pareto_df["方案编号"] = [f"P{i}" for i in range(1, len(pareto_df) + 1)]

    best_id = pareto_df.loc[0, "方案编号"]
    best_source = pareto[0]
    for item in pareto:
        ids = "、".join(item["metrics"]["selected_ids"])
        if ids == pareto_df.loc[0, "优选景点"]:
            best_source = item
            break

    itinerary_rows = []
    timeline_rows = []
    for day_idx, route_info in enumerate(best_source["routes"], start=1):
        route_ids = list(route_info["route"])
        names = spot.loc[route_ids, "name"].astype(str).tolist()
        itinerary_rows.append(
            {
                "日期": f"第{day_idx}天",
                "访问顺序": " -> ".join(route_ids),
                "景点名称": " -> ".join(names),
                "游览景点数": len(route_ids),
                "行车时间min": round(route_info["driving_minutes"], 1),
                "总活动耗时h": round(route_info["duration_hours"], 2),
                "预计回到酒店": time_to_text(route_info["end_time"]),
            }
        )
        for event in route_info["events"]:
            timeline_rows.append(
                {
                    "日期": f"第{day_idx}天",
                    "环节": event["环节"],
                    "地点": event["地点"],
                    "开始时间": time_to_text(event["开始"]),
                    "结束时间": time_to_text(event["结束"]),
                    "耗时h": round(event["结束"] - event["开始"], 2),
                }
            )

    itinerary_df = pd.DataFrame(itinerary_rows)
    timeline_df = pd.DataFrame(timeline_rows)
    selected_detail = spot.loc[best_source["metrics"]["selected_ids"]].reset_index()
    selected_detail = selected_detail[
        ["id", "name", "type", "preference", "comfort_time", "hotel_commute_min", "TOPSIS贴近度", "优先级"]
    ]
    selected_detail = selected_detail.rename(
        columns={
            "id": "景点ID",
            "name": "景点名称",
            "type": "类型",
            "preference": "喜好度",
            "comfort_time": "舒适游览时长h",
            "hotel_commute_min": "酒店车程min",
        }
    )
    pareto_df.insert(1, "是否最终基准方案", ["是" if code == best_id else "否" for code in pareto_df["方案编号"]])
    return pareto_df, itinerary_df, timeline_df, selected_detail


def save_tables(
    pareto_df: pd.DataFrame,
    itinerary_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    selected_detail: pd.DataFrame,
    history_df: pd.DataFrame,
) -> list[Path]:
    """Export CSV and XLSX tables for paper writing."""

    saved_files: list[Path] = []
    csv_outputs = [
        (PROCESSED_DIR / "problem2_pareto_solutions.csv", pareto_df),
        (PROCESSED_DIR / "problem2_baseline_itinerary.csv", itinerary_df),
        (PROCESSED_DIR / "problem2_baseline_timeline.csv", timeline_df),
        (PROCESSED_DIR / "problem2_selected_attractions.csv", selected_detail),
        (PROCESSED_DIR / "problem2_nsga2_history.csv", history_df),
    ]
    for path, table in csv_outputs:
        table.to_csv(path, index=False, encoding="utf-8-sig")
        saved_files.append(path)

    try:
        excel_path = TABLE_DIR / "problem2_model_outputs.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pareto_df.to_excel(writer, sheet_name="Pareto备选方案", index=False)
            itinerary_df.to_excel(writer, sheet_name="基准行程", index=False)
            timeline_df.to_excel(writer, sheet_name="详细时间轴", index=False)
            selected_detail.to_excel(writer, sheet_name="优选景点明细", index=False)
            history_df.to_excel(writer, sheet_name="收敛过程", index=False)
        saved_files.append(excel_path)
    except ModuleNotFoundError:
        print("提示：未安装 openpyxl，已跳过 Excel 输出；CSV 文件已正常保存。")

    return saved_files


def plot_results(pareto_df: pd.DataFrame, itinerary_df: pd.DataFrame, history_df: pd.DataFrame) -> None:
    """Create visualizations with titles, axes, legends and conclusion notes."""

    if not HAS_MATPLOTLIB:
        print("提示：未安装 matplotlib，已跳过可视化输出。")
        return

    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("问题二：无随机扰动下多目标景点优选与基准行程求解结果", fontsize=16, fontweight="bold")

    top = pareto_df.head(min(8, len(pareto_df))).copy()
    x = np.arange(len(top))
    axes[0, 0].bar(x - 0.18, top["总喜好度"], width=0.36, label="总喜好度", color="#4C78A8")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(top["方案编号"])
    axes[0, 0].set_title("Pareto备选方案满意度对比")
    axes[0, 0].set_xlabel("方案编号")
    axes[0, 0].set_ylabel("总喜好度")
    axes[0, 0].legend()
    best_pref = top.iloc[top["总喜好度"].argmax()]
    axes[0, 0].text(
        0.02,
        -0.24,
        f"结论：{best_pref['方案编号']}的总喜好度最高，为{best_pref['总喜好度']:.1f}。",
        transform=axes[0, 0].transAxes,
        fontsize=10,
    )

    axes[0, 1].bar(top["方案编号"], top["总行车时间min"], label="总行车时间", color="#F58518")
    axes[0, 1].set_title("Pareto备选方案行车负荷对比")
    axes[0, 1].set_xlabel("方案编号")
    axes[0, 1].set_ylabel("总行车时间/min")
    axes[0, 1].legend()
    best_drive = top.iloc[top["总行车时间min"].argmin()]
    axes[0, 1].text(
        0.02,
        -0.24,
        f"结论：{best_drive['方案编号']}行车负荷最低，总行车{best_drive['总行车时间min']:.0f}分钟。",
        transform=axes[0, 1].transAxes,
        fontsize=10,
    )

    axes[1, 0].plot(
        itinerary_df["日期"],
        itinerary_df["总活动耗时h"],
        marker="o",
        linewidth=2,
        label="每日总活动耗时",
        color="#54A24B",
    )
    axes[1, 0].bar(
        itinerary_df["日期"],
        itinerary_df["行车时间min"] / 60.0,
        alpha=0.45,
        label="每日行车时间",
        color="#B279A2",
    )
    axes[1, 0].set_title("最终基准行程每日负荷")
    axes[1, 0].set_xlabel("日期")
    axes[1, 0].set_ylabel("时间/h")
    axes[1, 0].legend()
    spread = itinerary_df["总活动耗时h"].max() - itinerary_df["总活动耗时h"].min()
    axes[1, 0].text(
        0.02,
        -0.26,
        f"结论：5天总活动耗时极差为{spread:.2f}小时，说明行程节奏较均衡。",
        transform=axes[1, 0].transAxes,
        fontsize=10,
    )

    axes[1, 1].plot(
        history_df["迭代代数"],
        history_df["当前最高喜好度"],
        label="最高喜好度",
        color="#4C78A8",
    )
    axes2 = axes[1, 1].twinx()
    axes2.plot(
        history_df["迭代代数"],
        history_df["当前最低行车时间min"],
        label="最低行车时间",
        color="#E45756",
    )
    axes[1, 1].set_title("搜索过程收敛趋势")
    axes[1, 1].set_xlabel("迭代代数")
    axes[1, 1].set_ylabel("喜好度")
    axes2.set_ylabel("行车时间/min")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, loc="best")
    axes[1, 1].text(
        0.02,
        -0.26,
        "结论：随着迭代推进，满意度和行车负荷逐步形成稳定的非支配方案集。",
        transform=axes[1, 1].transAxes,
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output = FIGURE_DIR / "problem2_model_visualization.png"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def print_solving_steps(cfg: Problem2Config) -> None:
    """Print paper-friendly solving-step explanation."""

    print("问题二求解步骤说明")
    print("=" * 60)
    print("步骤1 数据输入：读取问题一处理后的景点表、酒店车程、景点通勤矩阵、TOPSIS优先级结果。")
    print("注意事项：本脚本不重复问题一数据处理，若 CSV 缺失需先运行问题一代码。")
    print("步骤2 参数初始化：设置5天行程、每日1~2个景点、优选5~8个景点、双景点车程≤60分钟。")
    print(f"注意事项：综合筛选权重={cfg.final_weights}，三项权重和必须为1，可在[0,1]范围内调试。")
    print("步骤3 模型调用：生成可行日路线，采用遗传搜索和非支配排序得到Pareto备选方案集。")
    print("注意事项：若想提高搜索精度，可增大 population_size 和 generations，但运行时间会增加。")
    print("步骤4 结果输出：保存Pareto方案、最终5日基准行程、详细时间轴和可视化图。")
    print("注意事项：图表包含标题、坐标轴、图例和图下注释，可直接用于论文结果分析。")
    print("=" * 60)


def main() -> None:
    ensure_dirs()
    cfg = Problem2Config()
    print_solving_steps(cfg)

    print("\n步骤1：数据输入完成")
    attractions, hotel_commute, commute, _ = load_problem_data()
    print(f"景点数量：{len(attractions)}")

    print("\n步骤2：参数初始化完成")
    print(f"行程天数：{cfg.days}；优选景点数量范围：{cfg.min_spots}~{cfg.max_spots}")
    print(f"每日时间窗：{time_to_text(cfg.day_start)}-{time_to_text(cfg.day_end)}")
    print(f"种群规模：{cfg.population_size}；迭代次数：{cfg.generations}")

    print("\n步骤3：模型调用与多目标求解中...")
    pareto, aux, _ = solve_problem2(attractions, hotel_commute, commute, cfg)
    history_df = pd.DataFrame(aux["history"])
    print(f"可行日路线数量：{len(aux['day_options'])}")
    print(f"Pareto非支配方案数量：{len(pareto)}")

    print("\n步骤4：结果输出")
    pareto_df, itinerary_df, timeline_df, selected_detail = build_output_tables(pareto, attractions, cfg)
    saved_files = save_tables(pareto_df, itinerary_df, timeline_df, selected_detail, history_df)
    plot_results(pareto_df, itinerary_df, history_df)
    figure_path = FIGURE_DIR / "problem2_model_visualization.png"
    if figure_path.exists():
        saved_files.append(figure_path)

    best = pareto_df.iloc[0]
    print("\n最终基准方案摘要")
    print("-" * 60)
    print(f"方案编号：{best['方案编号']}")
    print(f"优选景点：{best['优选景点名称']}")
    print(f"总喜好度：{best['总喜好度']:.2f}")
    print(f"TOPSIS总贴近度：{best['TOPSIS总贴近度']:.3f}")
    print(f"总行车时间：{best['总行车时间min']:.0f} min")
    print(f"日耗时方差：{best['日耗时方差']:.4f}")
    print("\n5日基准行程：")
    print(itinerary_df.to_string(index=False))
    print("\n文件已保存：")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
