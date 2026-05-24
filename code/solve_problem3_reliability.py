"""
Problem 3 solver: stochastic disturbance reliability simulation.

Dependencies:
    pip install numpy pandas matplotlib openpyxl

Inputs:
    data/processed/problem2_baseline_timeline.csv
    data/processed/attractions_processed.csv

Outputs:
    data/processed/problem3_simulation_summary.csv
    data/processed/problem3_daily_risk.csv
    data/processed/problem3_disturbance_contribution.csv
    data/processed/problem3_detailed_results.csv
    results/figures/problem3_reliability_dashboard.png
    results/figures/problem3_risk_heatmap.png
    results/figures/problem3_risk_radar.png
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
class Problem3Config:
    n_sim: int = 10000
    scenario_n_sim: int = 3000
    improvement_n_sim: int = 3000
    seed: int = 20260523
    day_start: float = 7.0
    day_end_limit: float = 21.0
    reliability_target: float = 0.90
    # 初稿给出的是“发生延误后的额外耗时区间”。这里增加发生概率门控，
    # 避免把每个通勤/入园环节都视作必然遭遇最大扰动，更贴近随机模拟含义。
    traffic_peak_prob: float = 0.35
    traffic_off_prob: float = 0.15
    queue_peak_prob: float = 0.50
    queue_off_prob: float = 0.20


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def time_to_hour(value: str) -> float:
    hour, minute = str(value).split(":")
    return int(hour) + int(minute) / 60


def hour_to_text(value: float) -> str:
    minutes = int(round(value * 60))
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def is_traffic_peak(hour: float) -> bool:
    return any(start <= hour < end for start, end in [(7, 9), (11, 13), (16, 18)])


def is_queue_peak(hour: float) -> bool:
    return 9 <= hour < 12


def load_pre_data() -> pd.DataFrame:
    """Load and structure baseline timeline for simulation."""

    timeline_path = PROCESSED_DIR / "problem2_baseline_timeline.csv"
    attractions_path = PROCESSED_DIR / "attractions_processed.csv"
    if not timeline_path.exists() or not attractions_path.exists():
        raise FileNotFoundError("请先运行问题一和问题二代码，生成基准时间轴与景点基础数据。")

    timeline = pd.read_csv(timeline_path, encoding="utf-8-sig")
    attractions = pd.read_csv(attractions_path, encoding="utf-8-sig")
    spot_info = attractions[["id", "min_time", "effective_open_end"]].rename(
        columns={"id": "spot_id", "min_time": "min_visit_h", "effective_open_end": "close_h"}
    )

    timeline["base_start"] = timeline["开始时间"].map(time_to_hour)
    timeline["base_end"] = timeline["结束时间"].map(time_to_hour)
    timeline["base_dur"] = timeline["耗时h"].astype(float)
    timeline["spot_id"] = timeline.apply(
        lambda row: row["地点"] if row["环节"] == "景点游览" else None,
        axis=1,
    )
    timeline = timeline.merge(spot_info, on="spot_id", how="left")
    timeline["min_visit_h"] = timeline["min_visit_h"].fillna(0.0)
    timeline["close_h"] = timeline["close_h"].fillna(24.0)

    timeline["traffic_period"] = timeline["base_start"].map(lambda hour: "高峰" if is_traffic_peak(hour) else "平峰")
    timeline["queue_period"] = timeline["base_start"].map(lambda hour: "高峰" if is_queue_peak(hour) else "平峰")
    timeline["is_commute"] = timeline["环节"].str.contains("通勤|返程", regex=True)
    timeline["is_visit"] = timeline["环节"].eq("景点游览")

    timeline.to_csv(PROCESSED_DIR / "problem3_structured_timeline.csv", index=False, encoding="utf-8-sig")
    return timeline


def draw_delay(
    rng: np.random.Generator,
    occurs_prob: float,
    low: float,
    high: float,
) -> float:
    """Draw delay with a Bernoulli occurrence gate."""

    if rng.random() > occurs_prob:
        return 0.0
    return float(rng.uniform(low, high))


def simulate_once(base: pd.DataFrame, cfg: Problem3Config, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    """Run one Monte Carlo simulation and return summary plus per-day details."""

    all_pass = True
    visit_pass_all = True
    overtime_pass_all = True
    day_records: list[dict] = []
    traffic_delay_total = 0.0
    queue_delay_total = 0.0
    fail_days: list[str] = []

    for day, day_df in base.groupby("日期", sort=False):
        current = cfg.day_start
        day_traffic = 0.0
        day_queue = 0.0
        day_visit_fail = False
        day_overtime_fail = False
        weak_events: list[str] = []

        for _, row in day_df.iterrows():
            base_dur = float(row["base_dur"])
            traffic_delay = 0.0
            queue_delay = 0.0

            if bool(row["is_commute"]):
                if is_traffic_peak(current):
                    traffic_delay = draw_delay(rng, cfg.traffic_peak_prob, 1.0, 4.0)
                else:
                    traffic_delay = draw_delay(rng, cfg.traffic_off_prob, 0.0, 1.5)

            if bool(row["is_visit"]):
                if is_queue_peak(current):
                    queue_delay = draw_delay(rng, cfg.queue_peak_prob, 0.5, 3.0)
                else:
                    queue_delay = draw_delay(rng, cfg.queue_off_prob, 0.0, 1.0)

            actual_start = current
            actual_end = actual_start + base_dur + traffic_delay + queue_delay
            current = actual_end

            day_traffic += traffic_delay
            day_queue += queue_delay

            if bool(row["is_visit"]):
                # 按初稿假设：排队只推迟入园时刻，不压缩实际游览时长；
                # 因此游览达标重点检查舒适游览时长是否满足最小时长，以及推迟后是否晚于闭园。
                if base_dur < float(row["min_visit_h"]) or actual_end > float(row["close_h"]):
                    day_visit_fail = True
                    weak_events.append(f"{row['日期']}-{row['地点']}游览风险")

        if current > cfg.day_end_limit:
            day_overtime_fail = True
            weak_events.append(f"{day}返程超时")

        day_pass = not day_visit_fail and not day_overtime_fail
        if not day_pass:
            fail_days.append(day)
        all_pass = all_pass and day_pass
        visit_pass_all = visit_pass_all and (not day_visit_fail)
        overtime_pass_all = overtime_pass_all and (not day_overtime_fail)
        traffic_delay_total += day_traffic
        queue_delay_total += day_queue

        day_records.append(
            {
                "日期": day,
                "日结束时间": current,
                "道路延误h": day_traffic,
                "排队延误h": day_queue,
                "游览不达标": day_visit_fail,
                "当日超时": day_overtime_fail,
                "当日达标": day_pass,
                "薄弱事件": "；".join(weak_events) if weak_events else "无",
            }
        )

    if all_pass:
        fail_reason = "none"
    elif not visit_pass_all and not overtime_pass_all:
        fail_reason = "both"
    elif not visit_pass_all:
        fail_reason = "queue"
    else:
        fail_reason = "traffic"

    summary = {
        "是否达标": all_pass,
        "失败原因": fail_reason,
        "游览约束达标": visit_pass_all,
        "返程约束达标": overtime_pass_all,
        "道路总延误h": traffic_delay_total,
        "排队总延误h": queue_delay_total,
        "失败日期": "、".join(fail_days) if fail_days else "无",
    }
    return summary, day_records


def run_simulation(base: pd.DataFrame, cfg: Problem3Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo simulation and aggregate results."""

    rng = np.random.default_rng(cfg.seed)
    summary_rows = []
    day_rows = []
    convergence = []

    pass_count = 0
    for sim_id in range(1, cfg.n_sim + 1):
        summary, day_records = simulate_once(base, cfg, rng)
        summary["模拟编号"] = sim_id
        summary_rows.append(summary)
        pass_count += int(summary["是否达标"])
        for record in day_records:
            record["模拟编号"] = sim_id
            day_rows.append(record)
        if sim_id % 100 == 0:
            convergence.append({"模拟次数": sim_id, "累计可靠度": pass_count / sim_id})

    result = pd.DataFrame(summary_rows)
    day_detail = pd.DataFrame(day_rows)
    convergence_df = pd.DataFrame(convergence)

    fail = result[~result["是否达标"]]
    n_fail = len(fail)
    fail_counts = fail["失败原因"].value_counts()
    traffic_contrib = 0.0 if n_fail == 0 else (fail_counts.get("traffic", 0) + 0.5 * fail_counts.get("both", 0)) / n_fail
    queue_contrib = 0.0 if n_fail == 0 else (fail_counts.get("queue", 0) + 0.5 * fail_counts.get("both", 0)) / n_fail

    summary_table = pd.DataFrame(
        [
            ["整体可靠度", result["是否达标"].mean()],
            ["游览约束达标率", result["游览约束达标"].mean()],
            ["返程约束达标率", result["返程约束达标"].mean()],
            ["道路堵车贡献度", traffic_contrib],
            ["景点排队贡献度", queue_contrib],
            ["平均道路总延误h", result["道路总延误h"].mean()],
            ["平均排队总延误h", result["排队总延误h"].mean()],
        ],
        columns=["指标", "数值"],
    )

    daily = day_detail.groupby("日期").agg(
        每日失败率=("当日达标", lambda s: 1 - s.mean()),
        游览不达标率=("游览不达标", "mean"),
        当日超时率=("当日超时", "mean"),
        平均道路延误h=("道路延误h", "mean"),
        平均排队延误h=("排队延误h", "mean"),
        平均结束时间=("日结束时间", "mean"),
    ).reset_index()

    contribution = pd.DataFrame(
        {
            "扰动类型": ["道路堵车", "景点排队"],
            "失败贡献度": [traffic_contrib, queue_contrib],
            "平均延误h": [result["道路总延误h"].mean(), result["排队总延误h"].mean()],
        }
    )

    return result, day_detail, convergence_df, summary_table, daily, contribution


def run_scenario_grid(base: pd.DataFrame, cfg: Problem3Config) -> pd.DataFrame:
    """Evaluate reliability under all traffic/queue disturbance level combinations."""

    levels = [("低", 0.8), ("中", 1.0), ("高", 1.2)]
    rows: list[dict] = []
    for traffic_level, traffic_scale in levels:
        for queue_level, queue_scale in levels:
            scenario_cfg = Problem3Config(
                n_sim=cfg.scenario_n_sim,
                scenario_n_sim=cfg.scenario_n_sim,
                seed=cfg.seed,
                day_start=cfg.day_start,
                day_end_limit=cfg.day_end_limit,
                reliability_target=cfg.reliability_target,
                traffic_peak_prob=min(1.0, cfg.traffic_peak_prob * traffic_scale),
                traffic_off_prob=min(1.0, cfg.traffic_off_prob * traffic_scale),
                queue_peak_prob=min(1.0, cfg.queue_peak_prob * queue_scale),
                queue_off_prob=min(1.0, cfg.queue_off_prob * queue_scale),
            )
            scenario_result, _, _, _, _, _ = run_simulation(base, scenario_cfg)
            rows.append(
                {
                    "道路扰动水平": traffic_level,
                    "排队扰动水平": queue_level,
                    "道路高峰概率": scenario_cfg.traffic_peak_prob,
                    "道路平峰概率": scenario_cfg.traffic_off_prob,
                    "排队高峰概率": scenario_cfg.queue_peak_prob,
                    "排队平峰概率": scenario_cfg.queue_off_prob,
                    "整体可靠性": scenario_result["是否达标"].mean(),
                    "游览约束达标率": scenario_result["游览约束达标"].mean(),
                    "返程约束达标率": scenario_result["返程约束达标"].mean(),
                }
            )

    scenario_df = pd.DataFrame(rows)
    scenario_df["整体可靠性百分比"] = scenario_df["整体可靠性"] * 100
    return scenario_df


def make_single_spot_day(base: pd.DataFrame, day: str, spot_id: str) -> pd.DataFrame:
    """Create a robust-improvement timeline where one high-risk day keeps only one spot."""

    data = base.copy()
    other_days = data[data["日期"] != day]
    day_df = data[data["日期"] == day].copy()
    prep = day_df.iloc[[0]].copy()
    visit = day_df[(day_df["环节"] == "景点游览") & (day_df["地点"] == spot_id)].iloc[[0]].copy()

    original_commutes = day_df[day_df["is_commute"]].copy()
    hotel_to_spot = original_commutes[original_commutes["地点"].str.contains(f"酒店->{spot_id}", regex=False)]
    spot_to_hotel = original_commutes[original_commutes["地点"].str.contains(f"{spot_id}->酒店", regex=False)]

    if hotel_to_spot.empty and not spot_to_hotel.empty:
        hotel_to_spot = spot_to_hotel.iloc[[0]].copy()
        hotel_to_spot["环节"] = "酒店至景点通勤"
        hotel_to_spot["地点"] = f"酒店->{spot_id}"
    elif hotel_to_spot.empty:
        raise ValueError(f"无法构造 {day} 酒店到 {spot_id} 的通勤环节。")

    if spot_to_hotel.empty:
        spot_to_hotel = hotel_to_spot.iloc[[0]].copy()
        spot_to_hotel["环节"] = "返程至酒店"
        spot_to_hotel["地点"] = f"{spot_id}->酒店"

    rebuilt = pd.concat(
        [
            prep,
            hotel_to_spot.iloc[[0]],
            visit,
            spot_to_hotel.iloc[[0]],
        ],
        ignore_index=True,
    )
    result = pd.concat([other_days, rebuilt], ignore_index=True)
    return result


def compress_preparation_time(base: pd.DataFrame, days: set[str] | None = None, ratio: float = 0.8) -> pd.DataFrame:
    """Use the ±20% fixed-time flexibility to shorten preparation on selected days."""

    data = base.copy()
    mask = data["环节"].eq("起床早餐与整装")
    if days is not None:
        mask &= data["日期"].isin(days)
    data.loc[mask, "base_dur"] = data.loc[mask, "base_dur"].astype(float) * ratio
    data.loc[mask, "耗时h"] = data.loc[mask, "base_dur"]
    return data


def build_conservative_single_spot_plan() -> pd.DataFrame:
    """Build a 90%-target-oriented conservative plan: one spot per day with off-peak entry where possible."""

    attractions = pd.read_csv(PROCESSED_DIR / "attractions_processed.csv", encoding="utf-8-sig").set_index("id")
    hotel_commute = pd.read_csv(PROCESSED_DIR / "hotel_commute_minutes.csv", index_col=0, encoding="utf-8-sig")
    plan = [
        ("第1天", "A1", None),
        ("第2天", "A5", None),
        ("第3天", "A3", None),
        ("第4天", "A7", None),
        ("第5天", "A10", 12.0),
    ]

    rows: list[dict] = []
    for day, spot_id, target_visit_start in plan:
        spot = attractions.loc[spot_id]
        current = 7.0

        def append_event(label: str, location: str, duration: float, is_commute: bool, is_visit: bool) -> None:
            nonlocal current
            start = current
            end = current + duration
            rows.append(
                {
                    "日期": day,
                    "环节": label,
                    "地点": location,
                    "开始时间": hour_to_text(start),
                    "结束时间": hour_to_text(end),
                    "耗时h": duration,
                    "base_start": start,
                    "base_end": end,
                    "base_dur": duration,
                    "spot_id": spot_id if is_visit else None,
                    "min_visit_h": float(spot["min_time"]) if is_visit else 0.0,
                    "close_h": float(spot["effective_open_end"]) if is_visit else 24.0,
                    "traffic_period": "高峰" if is_traffic_peak(start) else "平峰",
                    "queue_period": "高峰" if is_queue_peak(start) else "平峰",
                    "is_commute": is_commute,
                    "is_visit": is_visit,
                }
            )
            current = end

        append_event("起床早餐与整装", "酒店", 1.2, False, False)
        commute_h = float(hotel_commute.loc["酒店", spot_id]) / 60.0
        append_event("酒店至景点通勤", f"酒店->{spot_id}", commute_h, True, False)

        desired_start = max(float(spot["effective_open_start"]), current)
        if target_visit_start is not None:
            desired_start = max(desired_start, target_visit_start)
        if desired_start > current:
            append_event("机动缓冲/错峰等待", f"{spot_id}周边", desired_start - current, False, False)

        append_event("景点游览", spot_id, float(spot["comfort_time"]), False, True)
        append_event("返程至酒店", f"{spot_id}->酒店", commute_h, True, False)
        append_event("正餐", "酒店/周边", 1.0, False, False)

    return pd.DataFrame(rows)


def run_robust_improvement_analysis(base: pd.DataFrame, cfg: Problem3Config) -> pd.DataFrame:
    """旁路模拟：评估若干不覆盖基准方案的稳健改进建议。"""

    # These plans are reported as side recommendations. The original P1 baseline
    # remains the comparison anchor so problem two outputs stay traceable.
    plans = [
        ("基准方案", "原问题二 P1 行程，不做调整。", base),
        ("全程准备压缩20%", "利用固定耗时±20%弹性，将每日起床早餐与整装由1.5h压缩为1.2h。", compress_preparation_time(base)),
        ("第5天仅游览A5", "将最高风险的第5天拆分为单景点日，仅保留民俗古村 A5，减少闭园压力。", make_single_spot_day(base, "第5天", "A5")),
        ("第5天仅游览A1", "将最高风险的第5天拆分为单景点日，仅保留古城老街 A1，减少连续游览风险。", make_single_spot_day(base, "第5天", "A1")),
        ("90%目标保守方案", "每日仅安排1个景点，优先选择全天开放或闭园较晚景点，并对A10采用12:00后错峰入园。", build_conservative_single_spot_plan()),
    ]

    rows: list[dict] = []
    for idx, (name, description, plan_base) in enumerate(plans):
        plan_cfg = Problem3Config(
            n_sim=cfg.improvement_n_sim,
            scenario_n_sim=cfg.scenario_n_sim,
            improvement_n_sim=cfg.improvement_n_sim,
            seed=cfg.seed + 1000 + idx,
            day_start=cfg.day_start,
            day_end_limit=cfg.day_end_limit,
            reliability_target=cfg.reliability_target,
            traffic_peak_prob=cfg.traffic_peak_prob,
            traffic_off_prob=cfg.traffic_off_prob,
            queue_peak_prob=cfg.queue_peak_prob,
            queue_off_prob=cfg.queue_off_prob,
        )
        result, _, _, summary, daily, _ = run_simulation(plan_base, plan_cfg)
        rows.append(
            {
                "改进方案": name,
                "方案说明": description,
                "模拟次数": plan_cfg.n_sim,
                "整体可靠度": float(summary.loc[summary["指标"] == "整体可靠度", "数值"].iloc[0]),
                "游览约束达标率": float(summary.loc[summary["指标"] == "游览约束达标率", "数值"].iloc[0]),
                "返程约束达标率": float(summary.loc[summary["指标"] == "返程约束达标率", "数值"].iloc[0]),
                "第5天失败率": float(daily.loc[daily["日期"] == "第5天", "每日失败率"].iloc[0]),
                "平均道路总延误h": float(result["道路总延误h"].mean()),
                "平均排队总延误h": float(result["排队总延误h"].mean()),
            }
        )

    comparison = pd.DataFrame(rows)
    baseline = float(comparison.loc[comparison["改进方案"] == "基准方案", "整体可靠度"].iloc[0])
    comparison["相对基准提升百分点"] = (comparison["整体可靠度"] - baseline) * 100
    comparison["整体可靠度百分比"] = comparison["整体可靠度"] * 100
    return comparison


def save_outputs(
    result: pd.DataFrame,
    day_detail: pd.DataFrame,
    convergence: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    contribution: pd.DataFrame,
    scenario_grid: pd.DataFrame,
    improvement: pd.DataFrame,
) -> list[Path]:
    outputs = [
        (PROCESSED_DIR / "problem3_detailed_results.csv", result),
        (PROCESSED_DIR / "problem3_daily_simulation_detail.csv", day_detail),
        (PROCESSED_DIR / "problem3_reliability_convergence.csv", convergence),
        (PROCESSED_DIR / "problem3_simulation_summary.csv", summary),
        (PROCESSED_DIR / "problem3_daily_risk.csv", daily),
        (PROCESSED_DIR / "problem3_disturbance_contribution.csv", contribution),
        (PROCESSED_DIR / "problem3_reliability_scenario_grid.csv", scenario_grid),
        (PROCESSED_DIR / "problem3_robust_improvement_comparison.csv", improvement),
    ]
    saved = []
    for path, table in outputs:
        table.to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(path)

    try:
        excel_path = TABLE_DIR / "problem3_reliability_outputs.xlsx"
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="summary", index=False)
            daily.to_excel(writer, sheet_name="daily_risk", index=False)
            contribution.to_excel(writer, sheet_name="contribution", index=False)
            scenario_grid.to_excel(writer, sheet_name="scenario_grid", index=False)
            improvement.to_excel(writer, sheet_name="robust_improvement", index=False)
            convergence.to_excel(writer, sheet_name="convergence", index=False)
            result.to_excel(writer, sheet_name="detailed_results", index=False)
        saved.append(excel_path)
    except ModuleNotFoundError:
        print("提示：未安装 openpyxl，已跳过 Excel 输出；CSV 文件已保存。")
    return saved


def plot_dashboard(
    convergence: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    contribution: pd.DataFrame,
    cfg: Problem3Config,
) -> Path | None:
    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    reliability = float(summary.loc[summary["指标"] == "整体可靠度", "数值"].iloc[0])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("问题三 行程可靠性评估结果可视化", fontsize=18, fontweight="bold")

    axes[0, 0].plot(convergence["模拟次数"], convergence["累计可靠度"], color="#2E86AB", linewidth=2.2)
    axes[0, 0].axhline(reliability, color="#A23B72", linestyle="--", label=f"最终可靠度={reliability:.3f}")
    axes[0, 0].axhline(cfg.reliability_target, color="#E63946", linestyle=":", label="目标可靠度=0.900")
    axes[0, 0].set_xlabel("模拟次数")
    axes[0, 0].set_ylabel("累计可靠度")
    axes[0, 0].set_title("Monte Carlo 可靠度收敛曲线")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    pass_rates = summary[summary["指标"].isin(["整体可靠度", "游览约束达标率", "返程约束达标率"])]
    bars = axes[0, 1].bar(pass_rates["指标"], pass_rates["数值"], color=["#2E86AB", "#2A9D8F", "#F4A261"])
    axes[0, 1].axhline(cfg.reliability_target, color="#E63946", linestyle="--", label="90%目标线")
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].set_ylabel("达标率")
    axes[0, 1].set_title("核心可靠性指标")
    axes[0, 1].legend()
    for bar in bars:
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}", ha="center")

    bars = axes[1, 0].bar(daily["日期"], daily["每日失败率"], color="#457B9D", alpha=0.8)
    axes[1, 0].set_xlabel("日期")
    axes[1, 0].set_ylabel("每日失败率")
    axes[1, 0].set_title("5日行程每日失败率对比")
    axes[1, 0].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{bar.get_height():.3f}", ha="center")

    bars = axes[1, 1].bar(contribution["扰动类型"], contribution["失败贡献度"], color=["#F18F01", "#C73E1D"], alpha=0.8)
    axes[1, 1].set_ylim(0, 1.05)
    axes[1, 1].set_ylabel("失败贡献度")
    axes[1, 1].set_title("两类扰动对行程失败的贡献度")
    axes[1, 1].grid(axis="y", alpha=0.3)
    for bar in bars:
        axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}", ha="center")

    output = FIGURE_DIR / "problem3_reliability_dashboard.png"
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_heatmap(daily: pd.DataFrame) -> Path | None:
    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    metrics = ["每日失败率", "游览不达标率", "当日超时率", "平均道路延误h", "平均排队延误h"]
    matrix = daily[metrics].to_numpy(dtype=float)
    # Normalize each metric for comparable color intensity.
    norm = matrix.copy()
    for j in range(norm.shape[1]):
        col = norm[:, j]
        span = col.max() - col.min()
        norm[:, j] = 0 if span == 0 else (col - col.min()) / span

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(norm, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=25, ha="right")
    ax.set_yticks(range(len(daily)))
    ax.set_yticklabels(daily["日期"])
    ax.set_title("问题三每日薄弱环节风险热力图", fontsize=15, fontweight="bold")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="#1F2937", fontsize=9)
    fig.colorbar(im, ax=ax, label="归一化风险强度")
    output = FIGURE_DIR / "problem3_risk_heatmap.png"
    fig.tight_layout()
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_radar(summary: pd.DataFrame, contribution: pd.DataFrame) -> Path | None:
    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    values = {
        "整体可靠度": float(summary.loc[summary["指标"] == "整体可靠度", "数值"].iloc[0]),
        "游览达标率": float(summary.loc[summary["指标"] == "游览约束达标率", "数值"].iloc[0]),
        "返程达标率": float(summary.loc[summary["指标"] == "返程约束达标率", "数值"].iloc[0]),
        "道路风险": float(contribution.loc[contribution["扰动类型"] == "道路堵车", "失败贡献度"].iloc[0]),
        "排队风险": float(contribution.loc[contribution["扰动类型"] == "景点排队", "失败贡献度"].iloc[0]),
    }
    labels = list(values.keys())
    vals = list(values.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
    ax.plot(angles, vals, color="#2A9D8F", linewidth=2)
    ax.fill(angles, vals, color="#2A9D8F", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("问题三可靠性与风险结构雷达图", fontsize=15, fontweight="bold", pad=20)
    output = FIGURE_DIR / "problem3_risk_radar.png"
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_scenario_heatmap(scenario_grid: pd.DataFrame) -> Path | None:
    """Plot the reliability range under all traffic/queue disturbance combinations."""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    traffic_order = ["低", "中", "高"]
    queue_order = ["低", "中", "高"]
    matrix = (
        scenario_grid.pivot(index="道路扰动水平", columns="排队扰动水平", values="整体可靠性百分比")
        .reindex(index=traffic_order, columns=queue_order)
    )

    fig, ax = plt.subplots(figsize=(8.8, 6.4))
    im = ax.imshow(matrix.values, cmap="RdYlGn", vmin=0, vmax=max(25, float(matrix.max().max())))
    ax.set_xticks(range(len(queue_order)))
    ax.set_xticklabels(queue_order)
    ax.set_yticks(range(len(traffic_order)))
    ax.set_yticklabels(traffic_order)
    ax.set_xlabel("景点排队扰动水平")
    ax.set_ylabel("道路堵车扰动水平")
    ax.set_title("问题三随机扰动组合下的可靠性范围", fontsize=15, fontweight="bold")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}%", ha="center", va="center", color="#111827", fontsize=11)

    fig.colorbar(im, ax=ax, label="整体可靠性/%")
    fig.text(
        0.08,
        0.02,
        f"关键结论：可靠性范围为 {matrix.min().min():.2f}% - {matrix.max().max():.2f}%，排队和堵车扰动越强，行程越不稳定。",
        fontsize=11,
        color="#334155",
    )
    output = FIGURE_DIR / "problem3_reliability_scenario_heatmap.png"
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_improvement_comparison(improvement: pd.DataFrame) -> Path | None:
    """Plot reliability comparison for robust-improvement suggestions."""

    if not HAS_MATPLOTLIB:
        return None
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    colors = ["#94A3B8", "#60A5FA", "#34D399", "#F59E0B", "#8B5CF6"]

    axes[0].bar(improvement["改进方案"], improvement["整体可靠度百分比"], color=colors, edgecolor="#334155")
    axes[0].axhline(90, color="#DC2626", linestyle="--", linewidth=1.5, label="90%可靠度目标")
    axes[0].set_ylabel("整体可靠度/%")
    axes[0].set_title("稳健改进建议的整体可靠度对比")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend()
    for idx, value in enumerate(improvement["整体可靠度百分比"]):
        axes[0].text(idx, value + 1.0, f"{value:.2f}%", ha="center", fontsize=9)

    axes[1].bar(improvement["改进方案"], improvement["第5天失败率"] * 100, color=colors, edgecolor="#334155")
    axes[1].set_ylabel("第5天失败率/%")
    axes[1].set_title("最高风险日改进效果对比")
    axes[1].tick_params(axis="x", rotation=18)
    for idx, value in enumerate(improvement["第5天失败率"] * 100):
        axes[1].text(idx, value + 1.0, f"{value:.2f}%", ha="center", fontsize=9)

    fig.suptitle("问题三稳健改进建议方案模拟", fontsize=16, fontweight="bold")
    fig.text(
        0.06,
        0.02,
        "关键结论：改进方案不替代问题二基准行程，仅用于说明薄弱点优化方向；拆分第5天双景点可显著降低最高风险日失败率。",
        fontsize=11,
        color="#334155",
    )
    output = FIGURE_DIR / "problem3_robust_improvement_comparison.png"
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(output, dpi=260, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    ensure_dirs()
    cfg = Problem3Config()
    base = load_pre_data()
    result, day_detail, convergence, summary, daily, contribution = run_simulation(base, cfg)
    scenario_grid = run_scenario_grid(base, cfg)
    improvement = run_robust_improvement_analysis(base, cfg)
    saved = save_outputs(result, day_detail, convergence, summary, daily, contribution, scenario_grid, improvement)
    for path in [
        plot_dashboard(convergence, summary, daily, contribution, cfg),
        plot_heatmap(daily),
        plot_radar(summary, contribution),
        plot_scenario_heatmap(scenario_grid),
        plot_improvement_comparison(improvement),
    ]:
        if path is not None:
            saved.append(path)

    print("问题三求解完成")
    print(summary.to_string(index=False))
    print("\n每日风险：")
    print(daily.round(4).to_string(index=False))
    print("\n随机扰动组合可靠性范围：")
    min_reliability = scenario_grid["整体可靠性百分比"].min()
    max_reliability = scenario_grid["整体可靠性百分比"].max()
    print(f"R ∈ [{min_reliability:.2f}%, {max_reliability:.2f}%]")
    print(scenario_grid[["道路扰动水平", "排队扰动水平", "整体可靠性百分比"]].round(2).to_string(index=False))
    print("\n稳健改进建议方案模拟：")
    print(improvement[["改进方案", "整体可靠度百分比", "相对基准提升百分点", "第5天失败率"]].round(2).to_string(index=False))
    print("\n文件已保存：")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
