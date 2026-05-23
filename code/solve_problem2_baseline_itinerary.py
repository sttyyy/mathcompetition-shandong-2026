"""
Problem 2 solver: multi-objective attraction selection and baseline itinerary.

依赖安装：
    pip install numpy pandas matplotlib openpyxl

本脚本直接调用问题一已经处理和建模后的数据，不重复数据预处理。

求解流程：
    1. 数据输入：读取景点信息、酒店车程、景点通勤矩阵、TOPSIS 优先级结果。
    2. 参数初始化：设置 5 日行程、每日景点数、通勤阈值、遗传搜索参数。
    3. 模型调用：生成可行日路线，使用遗传搜索和非支配排序得到 Pareto 备选方案。
    4. 结果输出：输出基准行程、详细时间轴、Pareto 方案、验证报告和可视化图。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    """集中管理参数，便于调试和论文说明。"""

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
    stable_start_generation: int = 30


def ensure_dirs() -> None:
    """创建输出目录。"""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """设置中文字体与简洁论文风图表样式。"""

    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def time_to_text(value: float) -> str:
    """将小数小时转换为 HH:MM。"""

    minutes = int(round(value * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def text_to_hour(value: str) -> float:
    """将 HH:MM 转换为小数小时。"""

    hour, minute = str(value).split(":")
    return int(hour) + int(minute) / 60


def minmax(values: Iterable[float], larger_is_better: bool = True) -> np.ndarray:
    """极差归一化，返回 [0,1] 区间数值。"""

    arr = np.asarray(list(values), dtype=float)
    span = arr.max() - arr.min()
    if math.isclose(float(span), 0.0):
        return np.ones_like(arr)
    scaled = (arr - arr.min()) / span
    return scaled if larger_is_better else 1.0 - scaled


def load_problem_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """步骤1：读取问题一输出数据。"""

    required = [
        PROCESSED_DIR / "attractions_processed.csv",
        PROCESSED_DIR / "hotel_commute_minutes.csv",
        PROCESSED_DIR / "attraction_commute_minutes.csv",
        PROCESSED_DIR / "problem1_latest_topsis_result.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少必要输入文件：\n" + "\n".join(missing))

    attractions = pd.read_csv(required[0], encoding="utf-8-sig")
    hotel_commute = pd.read_csv(required[1], index_col=0, encoding="utf-8-sig")
    commute = pd.read_csv(required[2], index_col=0, encoding="utf-8-sig")
    topsis = pd.read_csv(required[3], encoding="utf-8-sig")

    attractions = attractions.merge(
        topsis[["景点ID", "TOPSIS贴近度", "优先级"]],
        left_on="id",
        right_on="景点ID",
        how="left",
    ).drop(columns=["景点ID"])
    return attractions, hotel_commute, commute


def simulate_day(
    route: tuple[str, ...],
    attractions: pd.DataFrame,
    hotel_commute: pd.DataFrame,
    commute: pd.DataFrame,
    cfg: Problem2Config,
) -> dict | None:
    """构造单日时间轴，若违反开放时间或每日时间窗则返回 None。"""

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

    if len(route) == 1 and current_time <= cfg.day_end - cfg.meal_hours:
        meal_start = current_time
        current_time += cfg.meal_hours
        events.append({"环节": "正餐", "开始": meal_start, "结束": current_time, "地点": "酒店/周边"})

    if current_time > cfg.day_end:
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
    """枚举可行的单景点日路线和双景点日路线。"""

    ids = attractions["id"].tolist()
    options: list[dict] = []

    for spot_id in ids:
        result = simulate_day((spot_id,), attractions, hotel_commute, commute, cfg)
        if result is not None:
            options.append(result)

    for i in ids:
        for j in ids:
            if i == j or float(commute.loc[i, j]) > cfg.max_pair_commute_min:
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
    """修复重复景点和总景点数违规的个体。"""

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

    while total_spots(repaired) > cfg.max_spots:
        pair_indices = [idx for idx, opt in enumerate(repaired) if len(opt["route"]) == 2]
        if not pair_indices:
            break
        idx = rng.choice(pair_indices)
        used_without_day = {sid for k, opt in enumerate(repaired) if k != idx for sid in opt["route"]}
        candidates = [opt for opt in day_options if len(opt["route"]) == 1 and not (set(opt["route"]) & used_without_day)]
        if candidates:
            repaired[idx] = rng.choice(candidates)
        else:
            break

    while total_spots(repaired) < cfg.min_spots:
        single_indices = [idx for idx, opt in enumerate(repaired) if len(opt["route"]) == 1]
        if not single_indices:
            break
        idx = rng.choice(single_indices)
        used_without_day = {sid for k, opt in enumerate(repaired) if k != idx for sid in opt["route"]}
        candidates = [opt for opt in day_options if len(opt["route"]) == 2 and not (set(opt["route"]) & used_without_day)]
        if candidates:
            repaired[idx] = rng.choice(candidates)
        else:
            break
    return repaired


def random_individual(day_options: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """随机生成一个可行 5 日行程。"""

    for _ in range(2000):
        routes = rng.sample(day_options, cfg.days)
        repaired = repair_individual(routes, day_options, cfg, rng)
        selected = [sid for opt in repaired for sid in opt["route"]]
        if cfg.min_spots <= len(set(selected)) <= cfg.max_spots and len(set(selected)) == len(selected):
            return repaired
    raise RuntimeError("无法初始化可行行程，请检查约束。")


def evaluate(individual: list[dict], attractions: pd.DataFrame) -> dict:
    """计算三个目标函数：总喜好度、总行车时间、日耗时方差。"""

    spot = attractions.set_index("id")
    selected = [sid for opt in individual for sid in opt["route"]]
    daily_hours = np.array([opt["duration_hours"] for opt in individual], dtype=float)
    daily_drive = np.array([opt["driving_minutes"] for opt in individual], dtype=float)

    return {
        "selected_ids": selected,
        "selected_count": len(selected),
        "total_preference": float(spot.loc[selected, "preference"].sum()),
        "total_topsis": float(spot.loc[selected, "TOPSIS贴近度"].sum()),
        "total_drive_min": float(daily_drive.sum()),
        "balance_variance": float(np.mean((daily_hours - np.mean(daily_hours)) ** 2)),
        "daily_hours": daily_hours,
        "daily_drive": daily_drive,
    }


def dominates(a: dict, b: dict) -> bool:
    """判断 a 是否 Pareto 支配 b。"""

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
    """提取第一层非支配解。"""

    return [
        candidate
        for candidate in population
        if not any(dominates(other["metrics"], candidate["metrics"]) for other in population if other is not candidate)
    ]


def crossover(parent1: list[dict], parent2: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """单点交叉。"""

    point = rng.randint(1, cfg.days - 1)
    return parent1[:point] + parent2[point:]


def mutate(individual: list[dict], day_options: list[dict], cfg: Problem2Config, rng: random.Random) -> list[dict]:
    """随机替换某一天行程。"""

    mutated = list(individual)
    mutated[rng.randrange(cfg.days)] = rng.choice(day_options)
    return repair_individual(mutated, day_options, cfg, rng)


def solve_problem2(
    attractions: pd.DataFrame,
    hotel_commute: pd.DataFrame,
    commute: pd.DataFrame,
    cfg: Problem2Config,
) -> tuple[list[dict], dict]:
    """步骤3：使用遗传搜索和非支配排序求解。"""

    rng = random.Random(cfg.seed)
    day_options = build_day_options(attractions, hotel_commute, commute, cfg)
    population = [{"routes": random_individual(day_options, cfg, rng)} for _ in range(cfg.population_size)]
    for item in population:
        item["metrics"] = evaluate(item["routes"], attractions)

    history: list[dict] = []
    for generation in range(1, cfg.generations + 1):
        front = non_dominated_front(population)
        history.append(
            {
                "迭代代数": generation,
                "Pareto解数量": len(front),
                "当前最高喜好度": max(item["metrics"]["total_preference"] for item in front),
                "当前最低行车时间min": min(item["metrics"]["total_drive_min"] for item in front),
                "当前最低日耗时方差": min(item["metrics"]["balance_variance"] for item in front),
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

    return non_dominated_front(population), {"history": history, "day_options": day_options}


def score_for_final_choice(pareto: pd.DataFrame, cfg: Problem2Config) -> np.ndarray:
    """从 Pareto 解中筛选基准方案的综合得分。"""

    w1, w2, w3 = cfg.final_weights
    satisfaction = minmax(pareto["总喜好度"], larger_is_better=True)
    drive = minmax(pareto["总行车时间min"], larger_is_better=False)
    balance = minmax(pareto["日耗时方差"], larger_is_better=False)
    return w1 * satisfaction + w2 * drive + w3 * balance


def build_output_tables(
    pareto: list[dict],
    attractions: pd.DataFrame,
    cfg: Problem2Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """整理 Pareto 方案表、基准行程表、时间轴表和入选景点明细。"""

    spot = attractions.set_index("id")
    rows = []
    for idx, item in enumerate(pareto, start=1):
        metrics = item["metrics"]
        rows.append(
            {
                "原始方案编号": f"R{idx}",
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
    pareto_df.insert(1, "是否最终基准方案", ["是" if i == 0 else "否" for i in range(len(pareto_df))])

    best_key = pareto_df.loc[0, "优选景点"]
    best_source = next((item for item in pareto if "、".join(item["metrics"]["selected_ids"]) == best_key), pareto[0])

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

    selected_detail = spot.loc[best_source["metrics"]["selected_ids"]].reset_index()
    selected_detail = selected_detail[
        ["id", "name", "type", "preference", "comfort_time", "hotel_commute_min", "TOPSIS贴近度", "优先级"]
    ].rename(
        columns={
            "id": "景点ID",
            "name": "景点名称",
            "type": "类型",
            "preference": "喜好度",
            "comfort_time": "舒适游览时长h",
            "hotel_commute_min": "酒店车程min",
        }
    )
    return pareto_df, pd.DataFrame(itinerary_rows), pd.DataFrame(timeline_rows), selected_detail, best_source


def validate_solution(
    itinerary_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    pareto_df: pd.DataFrame,
    history_df: pd.DataFrame,
    cfg: Problem2Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """新增验证环节：可行性、收敛性和多目标折中检验。"""

    route_ids = []
    for route_text in itinerary_df["访问顺序"]:
        route_ids.extend([part.strip() for part in route_text.split("->")])

    end_hours = timeline_df.groupby("日期")["结束时间"].max().map(text_to_hour)
    latest_end = float(end_hours.max())
    stable = history_df[history_df["迭代代数"] >= cfg.stable_start_generation]
    convergence_ok = (
        stable["当前最高喜好度"].nunique() <= 2
        and stable["当前最低行车时间min"].nunique() <= 2
        and stable["当前最低日耗时方差"].nunique() <= 2
    )
    best = pareto_df.iloc[0]
    validation_rows = [
        {
            "检验项目": "优选景点数量约束",
            "判定标准": f"{cfg.min_spots} <= 优选景点数 <= {cfg.max_spots}",
            "实际结果": len(route_ids),
            "是否通过": "通过" if cfg.min_spots <= len(route_ids) <= cfg.max_spots else "不通过",
        },
        {
            "检验项目": "景点不重复约束",
            "判定标准": "每个景点最多游览一次",
            "实际结果": f"总访问{len(route_ids)}个，唯一景点{len(set(route_ids))}个",
            "是否通过": "通过" if len(route_ids) == len(set(route_ids)) else "不通过",
        },
        {
            "检验项目": "每日景点数量约束",
            "判定标准": "每日1-2个景点",
            "实际结果": f"范围{itinerary_df['游览景点数'].min()}-{itinerary_df['游览景点数'].max()}个",
            "是否通过": "通过" if itinerary_df["游览景点数"].between(1, 2).all() else "不通过",
        },
        {
            "检验项目": "每日时间窗约束",
            "判定标准": "每日21:00前返回酒店",
            "实际结果": f"最晚{time_to_text(latest_end)}",
            "是否通过": "通过" if latest_end <= cfg.day_end else "不通过",
        },
        {
            "检验项目": "收敛性检验",
            "判定标准": f"第{cfg.stable_start_generation}代后核心目标基本稳定",
            "实际结果": "稳定" if convergence_ok else "仍有波动",
            "是否通过": "通过" if convergence_ok else "需关注",
        },
        {
            "检验项目": "多目标折中检验",
            "判定标准": "综合筛选得分排名第一",
            "实际结果": f"{best['方案编号']} 得分 {best['综合筛选得分']:.3f}",
            "是否通过": "通过",
        },
    ]
    validation_df = pd.DataFrame(validation_rows)

    stable_summary = pd.DataFrame(
        [
            {
                "阶段": f"第{cfg.stable_start_generation}代以后",
                "最高喜好度极差": stable["当前最高喜好度"].max() - stable["当前最高喜好度"].min(),
                "最低行车时间极差min": stable["当前最低行车时间min"].max() - stable["当前最低行车时间min"].min(),
                "最低日耗时方差极差": stable["当前最低日耗时方差"].max() - stable["当前最低日耗时方差"].min(),
                "Pareto解数量末值": int(history_df.iloc[-1]["Pareto解数量"]),
            }
        ]
    )
    return validation_df, stable_summary


def save_tables(
    pareto_df: pd.DataFrame,
    itinerary_df: pd.DataFrame,
    timeline_df: pd.DataFrame,
    selected_detail: pd.DataFrame,
    history_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    convergence_summary: pd.DataFrame,
) -> list[Path]:
    """保存所有表格。"""

    def write_csv_safely(path: Path, table: pd.DataFrame) -> Path:
        try:
            table.to_csv(path, index=False, encoding="utf-8-sig")
            return path
        except PermissionError:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fallback = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            table.to_csv(fallback, index=False, encoding="utf-8-sig")
            print(f"提示：{path.name} 可能正被 Excel/WPS 打开，已另存为 {fallback.name}")
            return fallback

    outputs = [
        (PROCESSED_DIR / "problem2_pareto_solutions.csv", pareto_df),
        (PROCESSED_DIR / "problem2_baseline_itinerary.csv", itinerary_df),
        (PROCESSED_DIR / "problem2_baseline_timeline.csv", timeline_df),
        (PROCESSED_DIR / "problem2_selected_attractions.csv", selected_detail),
        (PROCESSED_DIR / "problem2_nsga2_history.csv", history_df),
        (PROCESSED_DIR / "problem2_validation_report.csv", validation_df),
        (PROCESSED_DIR / "problem2_convergence_summary.csv", convergence_summary),
    ]
    saved_files: list[Path] = []
    for path, table in outputs:
        saved_files.append(write_csv_safely(path, table))

    try:
        excel_path = TABLE_DIR / "problem2_model_outputs.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            pareto_df.to_excel(writer, sheet_name="Pareto备选方案", index=False)
            itinerary_df.to_excel(writer, sheet_name="基准行程", index=False)
            timeline_df.to_excel(writer, sheet_name="详细时间轴", index=False)
            selected_detail.to_excel(writer, sheet_name="优选景点明细", index=False)
            history_df.to_excel(writer, sheet_name="收敛过程", index=False)
            validation_df.to_excel(writer, sheet_name="验证报告", index=False)
            convergence_summary.to_excel(writer, sheet_name="收敛摘要", index=False)
        saved_files.append(excel_path)
    except ModuleNotFoundError:
        print("提示：未安装 openpyxl，已跳过 Excel 输出；CSV 文件已正常保存。")
    return saved_files


def add_best_value_line(ax, value: float, label: str, color: str) -> None:
    """给收敛曲线添加最优值水平线。"""

    ax.axhline(value, color=color, linestyle=":", linewidth=1, label=label)
    ax.legend(fontsize=9)


def plot_convergence(history_df: pd.DataFrame, cfg: Problem2Config) -> Path | None:
    """绘制 NSGA-II 收敛四宫格。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("NSGA-II 多目标优化收敛过程分析", fontsize=16, fontweight="bold")
    x = history_df["迭代代数"]

    axes[0, 0].plot(x, history_df["当前最高喜好度"], color="#E63946", marker="o", markersize=2)
    axes[0, 0].set_title("(a) 最高喜好度收敛曲线")
    axes[0, 0].set_xlabel("迭代代数")
    axes[0, 0].set_ylabel("最高喜好度")
    add_best_value_line(axes[0, 0], history_df["当前最高喜好度"].max(), f"最优值={history_df['当前最高喜好度'].max():.2f}", "#E63946")

    axes[0, 1].plot(x, history_df["当前最低行车时间min"], color="#2A9D8F", marker="o", markersize=2)
    axes[0, 1].set_title("(b) 最低行车时间收敛曲线")
    axes[0, 1].set_xlabel("迭代代数")
    axes[0, 1].set_ylabel("最低行车时间/min")
    add_best_value_line(axes[0, 1], history_df["当前最低行车时间min"].min(), f"最优值={history_df['当前最低行车时间min'].min():.0f} min", "#2A9D8F")

    axes[1, 0].plot(x, history_df["当前最低日耗时方差"], color="#264653", marker="o", markersize=2)
    axes[1, 0].set_title("(c) 最低日耗时方差收敛曲线")
    axes[1, 0].set_xlabel("迭代代数")
    axes[1, 0].set_ylabel("最低日耗时方差")
    add_best_value_line(axes[1, 0], history_df["当前最低日耗时方差"].min(), f"最优值={history_df['当前最低日耗时方差'].min():.4f}", "#264653")

    axes[1, 1].plot(x, history_df["Pareto解数量"], color="#E9C46A", marker="o", markersize=2)
    axes[1, 1].set_title("(d) Pareto 前沿解数量收敛曲线")
    axes[1, 1].set_xlabel("迭代代数")
    axes[1, 1].set_ylabel("Pareto解数量")
    add_best_value_line(axes[1, 1], history_df["Pareto解数量"].iloc[-1], f"最终数量={history_df['Pareto解数量'].iloc[-1]:.0f}", "#E9C46A")

    for ax in axes.ravel():
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.axvspan(cfg.stable_start_generation, cfg.generations, color="#E5E5E5", alpha=0.45)

    output = FIGURE_DIR / "problem2_convergence_analysis.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_normalized_convergence(history_df: pd.DataFrame, cfg: Problem2Config) -> Path | None:
    """绘制归一化收敛趋势对比图。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = history_df["迭代代数"]
    pref = minmax(history_df["当前最高喜好度"], larger_is_better=True)
    drive = minmax(history_df["当前最低行车时间min"], larger_is_better=False)
    balance = minmax(history_df["当前最低日耗时方差"], larger_is_better=False)

    ax.plot(x, pref, color="#E63946", linewidth=2, label="喜好度（越大越好）")
    ax.plot(x, drive, color="#2A9D8F", linewidth=2, label="行车负荷（越小越好）")
    ax.plot(x, balance, color="#264653", linewidth=2, label="日耗时方差（越小越好）")
    ax.axvspan(cfg.stable_start_generation, cfg.generations, color="#E5E5E5", alpha=0.5, label="稳定区间")
    ax.set_title("多目标归一化收敛趋势对比", fontsize=15, fontweight="bold")
    ax.set_xlabel("迭代代数")
    ax.set_ylabel("归一化指标（0=最差，1=最优）")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    output = FIGURE_DIR / "problem2_normalized_convergence.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_pareto_score_ranking(pareto_df: pd.DataFrame) -> Path | None:
    """绘制 Pareto 方案综合得分排名。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    top = pareto_df.head(min(15, len(pareto_df))).iloc[::-1]
    colors = ["#E63946" if flag == "是" else "#4C78A8" for flag in top["是否最终基准方案"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(top["方案编号"], top["综合筛选得分"], color=colors, edgecolor="#333333")
    for y, value in zip(top["方案编号"], top["综合筛选得分"]):
        ax.text(value + 0.005, y, f"{value:.3f}", va="center", fontsize=9)
    ax.set_title("Pareto 方案综合得分排名（前15）", fontsize=15, fontweight="bold")
    ax.set_xlabel("综合筛选得分")
    ax.set_ylabel("方案编号")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    output = FIGURE_DIR / "problem2_pareto_score_ranking.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_pareto_front_2d(pareto_df: pd.DataFrame) -> Path | None:
    """绘制二维 Pareto 前沿散点图。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    best = pareto_df.iloc[0]
    sizes = 80 + 260 * minmax(pareto_df["综合筛选得分"], larger_is_better=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        pareto_df["总喜好度"],
        pareto_df["总行车时间min"],
        s=sizes,
        c=pareto_df["日耗时方差"],
        cmap="viridis",
        alpha=0.78,
        edgecolor="#333333",
        label="其他方案",
    )
    ax.scatter(
        best["总喜好度"],
        best["总行车时间min"],
        s=260,
        color="red",
        edgecolor="black",
        marker="*",
        label=f"基准方案 {best['方案编号']}",
        zorder=5,
    )
    ax.set_title("Pareto 前沿：总喜好度 vs 总行车时间\n点大小∝综合筛选得分，颜色∝日耗时方差", fontsize=14, fontweight="bold")
    ax.set_xlabel("总喜好度")
    ax.set_ylabel("总行车时间/min")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="日耗时方差")

    output = FIGURE_DIR / "problem2_pareto_front_2d.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_pareto_front_3d(pareto_df: pd.DataFrame) -> Path | None:
    """绘制三维 Pareto 前沿图。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    best = pareto_df.iloc[0]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        pareto_df["总喜好度"],
        pareto_df["总行车时间min"],
        pareto_df["日耗时方差"],
        c="#4C78A8",
        alpha=0.65,
        label="其他方案",
    )
    ax.scatter(
        best["总喜好度"],
        best["总行车时间min"],
        best["日耗时方差"],
        c="red",
        marker="*",
        s=220,
        edgecolor="black",
        label=f"基准方案 {best['方案编号']}",
    )
    ax.set_title("Pareto 三维前沿分布", fontsize=15, fontweight="bold")
    ax.set_xlabel("总喜好度")
    ax.set_ylabel("总行车时间/min")
    ax.set_zlabel("日耗时方差")
    ax.legend()

    output = FIGURE_DIR / "problem2_pareto_front_3d.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_overview(pareto_df: pd.DataFrame, itinerary_df: pd.DataFrame, history_df: pd.DataFrame) -> Path | None:
    """保留一张总览图，便于快速汇报。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("问题二：多目标景点优选与基准行程求解结果", fontsize=16, fontweight="bold")

    top = pareto_df.head(min(8, len(pareto_df)))
    axes[0, 0].bar(top["方案编号"], top["总喜好度"], label="总喜好度", color="#4C78A8")
    axes[0, 0].set_title("Pareto备选方案满意度对比")
    axes[0, 0].set_xlabel("方案编号")
    axes[0, 0].set_ylabel("总喜好度")
    axes[0, 0].legend()

    axes[0, 1].bar(top["方案编号"], top["总行车时间min"], label="总行车时间", color="#F58518")
    axes[0, 1].set_title("Pareto备选方案行车负荷对比")
    axes[0, 1].set_xlabel("方案编号")
    axes[0, 1].set_ylabel("总行车时间/min")
    axes[0, 1].legend()

    axes[1, 0].plot(itinerary_df["日期"], itinerary_df["总活动耗时h"], marker="o", label="每日总活动耗时", color="#54A24B")
    axes[1, 0].bar(itinerary_df["日期"], itinerary_df["行车时间min"] / 60.0, alpha=0.45, label="每日行车时间", color="#B279A2")
    axes[1, 0].set_title("最终基准行程每日负荷")
    axes[1, 0].set_xlabel("日期")
    axes[1, 0].set_ylabel("时间/h")
    axes[1, 0].legend()

    axes[1, 1].plot(history_df["迭代代数"], history_df["当前最高喜好度"], label="最高喜好度", color="#4C78A8")
    axes2 = axes[1, 1].twinx()
    axes2.plot(history_df["迭代代数"], history_df["当前最低行车时间min"], label="最低行车时间", color="#E45756")
    axes[1, 1].set_title("搜索过程收敛趋势")
    axes[1, 1].set_xlabel("迭代代数")
    axes[1, 1].set_ylabel("喜好度")
    axes2.set_ylabel("行车时间/min")
    lines, labels = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = axes2.get_legend_handles_labels()
    axes[1, 1].legend(lines + lines2, labels + labels2, loc="best")

    output = FIGURE_DIR / "problem2_model_visualization.png"
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_radar_comparison(pareto_df: pd.DataFrame) -> Path | None:
    """绘制代表性 Pareto 方案雷达图，展示三目标折中关系。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()

    candidates = [
        ("基准方案", pareto_df.iloc[0]),
        ("最高满意度", pareto_df.loc[pareto_df["总喜好度"].idxmax()]),
        ("最低行车", pareto_df.loc[pareto_df["总行车时间min"].idxmin()]),
        ("最均衡", pareto_df.loc[pareto_df["日耗时方差"].idxmin()]),
    ]
    metrics = [
        ("满意度", "总喜好度", True),
        ("优先级", "TOPSIS总贴近度", True),
        ("行车轻量", "总行车时间min", False),
        ("日程均衡", "日耗时方差", False),
        ("综合得分", "综合筛选得分", True),
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw={"polar": True})
    colors = ["#E63946", "#457B9D", "#2A9D8F", "#F4A261"]
    for (label, row), color in zip(candidates, colors):
        values = []
        for _, column, larger_is_better in metrics:
            score = pd.Series(minmax(pareto_df[column], larger_is_better=larger_is_better), index=pareto_df.index)
            row_score = float(score.loc[row.name])
            values.append(row_score)
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=f"{label} {row['方案编号']}")
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([name for name, _, _ in metrics], fontsize=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1.05)
    ax.set_title("代表性 Pareto 方案多目标雷达对比", fontsize=15, fontweight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12))
    fig.text(
        0.08,
        0.04,
        "关键结论：雷达图越外扩表示该目标表现越优，基准方案是在满意度、行车负荷与日程均衡之间的折中解。",
        fontsize=11,
        color="#334155",
    )

    output = FIGURE_DIR / "problem2_radar_comparison.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_daily_load_heatmap(itinerary_df: pd.DataFrame) -> Path | None:
    """绘制最终基准行程的每日负荷热力图。"""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()

    columns = ["游览景点数", "行车时间min", "总活动耗时h"]
    matrix = itinerary_df.set_index("日期")[columns].astype(float)
    normalized = matrix.copy()
    for column in normalized.columns:
        span = normalized[column].max() - normalized[column].min()
        normalized[column] = 0.0 if span == 0 else (normalized[column] - normalized[column].min()) / span

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    image = ax.imshow(normalized.values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(["景点数", "行车时间", "活动耗时"], fontsize=11)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=11)
    ax.set_title("最终基准行程每日负荷热力图", fontsize=15, fontweight="bold", pad=14)

    for i, day in enumerate(matrix.index):
        for j, column in enumerate(columns):
            value = matrix.loc[day, column]
            suffix = "个" if column == "游览景点数" else ("min" if column == "行车时间min" else "h")
            ax.text(j, i, f"{value:.1f}{suffix}", ha="center", va="center", color="#0F172A", fontsize=10)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("归一化负荷强度", rotation=270, labelpad=16)
    fig.text(
        0.08,
        0.02,
        "关键结论：颜色越深表示当天相对负荷越高，可用于识别需要缓冲时间的日期。",
        fontsize=11,
        color="#334155",
    )

    output = FIGURE_DIR / "problem2_daily_load_heatmap.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_all_results(pareto_df: pd.DataFrame, itinerary_df: pd.DataFrame, history_df: pd.DataFrame, cfg: Problem2Config) -> list[Path]:
    """生成所有可视化图。"""

    paths = [
        plot_overview(pareto_df, itinerary_df, history_df),
        plot_convergence(history_df, cfg),
        plot_normalized_convergence(history_df, cfg),
        plot_pareto_score_ranking(pareto_df),
        plot_pareto_front_2d(pareto_df),
        plot_pareto_front_3d(pareto_df),
        plot_radar_comparison(pareto_df),
        plot_daily_load_heatmap(itinerary_df),
    ]
    return [path for path in paths if path is not None]


def print_solving_steps(cfg: Problem2Config) -> None:
    """打印求解步骤与注意事项。"""

    print("问题二求解步骤说明")
    print("=" * 60)
    print("步骤1 数据输入：读取问题一处理后的景点表、酒店车程、景点通勤矩阵、TOPSIS优先级结果。")
    print("注意事项：本脚本不重复问题一数据处理，若 CSV 缺失需先运行问题一代码。")
    print("步骤2 参数初始化：设置5天行程、每日1-2个景点、优选5-8个景点、双景点车程≤60分钟。")
    print(f"注意事项：综合筛选权重={cfg.final_weights}，三项权重和必须为1，可在[0,1]范围内调试。")
    print("步骤3 模型调用：生成可行日路线，采用遗传搜索和非支配排序得到Pareto备选方案集。")
    print("注意事项：若想提高搜索精度，可增大 population_size 和 generations，但运行时间会增加。")
    print("步骤4 结果输出与验证：保存方案表、行程表、时间轴、验证报告和多类可视化图。")
    print("注意事项：验证包括可行性、收敛性和多目标折中合理性。")
    print("=" * 60)


def main() -> None:
    ensure_dirs()
    cfg = Problem2Config()
    print_solving_steps(cfg)

    print("\n步骤1：数据输入完成")
    attractions, hotel_commute, commute = load_problem_data()
    print(f"景点数量：{len(attractions)}")

    print("\n步骤2：参数初始化完成")
    print(f"行程天数：{cfg.days}；优选景点数量范围：{cfg.min_spots}~{cfg.max_spots}")
    print(f"每日时间窗：{time_to_text(cfg.day_start)}-{time_to_text(cfg.day_end)}")
    print(f"种群规模：{cfg.population_size}；迭代次数：{cfg.generations}")

    print("\n步骤3：模型调用与多目标求解中...")
    pareto, aux = solve_problem2(attractions, hotel_commute, commute, cfg)
    history_df = pd.DataFrame(aux["history"])
    print(f"可行日路线数量：{len(aux['day_options'])}")
    print(f"Pareto非支配方案数量：{len(pareto)}")

    print("\n步骤4：结果输出与验证")
    pareto_df, itinerary_df, timeline_df, selected_detail, _ = build_output_tables(pareto, attractions, cfg)
    validation_df, convergence_summary = validate_solution(itinerary_df, timeline_df, pareto_df, history_df, cfg)
    saved_files = save_tables(
        pareto_df,
        itinerary_df,
        timeline_df,
        selected_detail,
        history_df,
        validation_df,
        convergence_summary,
    )
    figure_files = plot_all_results(pareto_df, itinerary_df, history_df, cfg)
    saved_files.extend(figure_files)

    best = pareto_df.iloc[0]
    print("\n最终基准方案摘要")
    print("-" * 60)
    print(f"方案编号：{best['方案编号']}")
    print(f"优选景点：{best['优选景点名称']}")
    print(f"总喜好度：{best['总喜好度']:.2f}")
    print(f"TOPSIS总贴近度：{best['TOPSIS总贴近度']:.3f}")
    print(f"总行车时间：{best['总行车时间min']:.0f} min")
    print(f"日耗时方差：{best['日耗时方差']:.4f}")

    print("\n验证报告：")
    print(validation_df.to_string(index=False))
    print("\n5日基准行程：")
    print(itinerary_df.to_string(index=False))
    print("\n文件已保存：")
    for path in saved_files:
        print(f"- {path}")


if __name__ == "__main__":
    main()
