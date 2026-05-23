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


def save_outputs(
    result: pd.DataFrame,
    day_detail: pd.DataFrame,
    convergence: pd.DataFrame,
    summary: pd.DataFrame,
    daily: pd.DataFrame,
    contribution: pd.DataFrame,
) -> list[Path]:
    outputs = [
        (PROCESSED_DIR / "problem3_detailed_results.csv", result),
        (PROCESSED_DIR / "problem3_daily_simulation_detail.csv", day_detail),
        (PROCESSED_DIR / "problem3_reliability_convergence.csv", convergence),
        (PROCESSED_DIR / "problem3_simulation_summary.csv", summary),
        (PROCESSED_DIR / "problem3_daily_risk.csv", daily),
        (PROCESSED_DIR / "problem3_disturbance_contribution.csv", contribution),
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


def main() -> None:
    ensure_dirs()
    cfg = Problem3Config()
    base = load_pre_data()
    result, day_detail, convergence, summary, daily, contribution = run_simulation(base, cfg)
    saved = save_outputs(result, day_detail, convergence, summary, daily, contribution)
    for path in [
        plot_dashboard(convergence, summary, daily, contribution, cfg),
        plot_heatmap(daily),
        plot_radar(summary, contribution),
    ]:
        if path is not None:
            saved.append(path)

    print("问题三求解完成")
    print(summary.to_string(index=False))
    print("\n每日风险：")
    print(daily.round(4).to_string(index=False))
    print("\n文件已保存：")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
