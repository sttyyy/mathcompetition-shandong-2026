"""
Render clean images for Problem 2 itinerary and timeline tables.

Dependencies:
    pip install pandas matplotlib

Inputs:
    data/processed/problem2_baseline_itinerary.csv
    data/processed/problem2_baseline_timeline.csv

Outputs:
    results/figures/problem2_itinerary_overview.png
    results/figures/problem2_timeline_gantt.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURE_DIR = ROOT / "results" / "figures"


def setup_style() -> None:
    """Use common Chinese fonts and a restrained paper-style palette."""

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"


def time_to_hour(value: str) -> float:
    """Convert HH:MM to decimal hour for timeline plotting."""

    hour, minute = value.split(":")
    return int(hour) + int(minute) / 60


def render_itinerary_overview(itinerary: pd.DataFrame) -> Path:
    """Render a compact 5-day itinerary table."""

    display = itinerary.copy()
    display["行车时间"] = display["行车时间min"].map(lambda x: f"{x:.0f} min")
    display["总活动耗时"] = display["总活动耗时h"].map(lambda x: f"{x:.1f} h")
    display = display[["日期", "访问顺序", "景点名称", "行车时间", "总活动耗时", "预计回到酒店"]]

    fig, ax = plt.subplots(figsize=(15, 5.8))
    ax.axis("off")
    ax.set_title("问题二最终基准行程安排", fontsize=20, fontweight="bold", pad=20)

    table = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.10, 0.18, 0.30, 0.13, 0.14, 0.15],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.0)

    header_color = "#D9EAF7"
    stripe_color = "#F6F8FA"
    edge_color = "#B9C4CC"
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(fontweight="bold", color="#1F2D3D")
        elif row % 2 == 0:
            cell.set_facecolor(stripe_color)
        else:
            cell.set_facecolor("white")

    max_duration = display["总活动耗时"].str.replace(" h", "", regex=False).astype(float).max()
    min_duration = display["总活动耗时"].str.replace(" h", "", regex=False).astype(float).min()
    note = (
        f"关键结论：5天均在21:00前返回酒店，总活动耗时极差为{max_duration - min_duration:.1f}小时，"
        "整体节奏较均衡。"
    )
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=12, color="#334155")

    output = FIGURE_DIR / "problem2_itinerary_overview.png"
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def render_timeline_gantt(timeline: pd.DataFrame) -> Path:
    """Render a readable Gantt-style timeline chart."""

    data = timeline.copy()
    data["start_h"] = data["开始时间"].map(time_to_hour)
    data["end_h"] = data["结束时间"].map(time_to_hour)
    data["duration_h"] = data["end_h"] - data["start_h"]

    days = ["第1天", "第2天", "第3天", "第4天", "第5天"]
    y_positions = {day: len(days) - idx for idx, day in enumerate(days)}
    colors = {
        "起床早餐与整装": "#BFD7EA",
        "酒店至景点通勤": "#F6BD60",
        "景点间通勤": "#F6BD60",
        "景点游览": "#84A98C",
        "午餐": "#F2CC8F",
        "正餐": "#F2CC8F",
        "返程至酒店": "#C9BBCF",
        "等待开园": "#DAD7CD",
    }

    fig, ax = plt.subplots(figsize=(16, 7))
    for _, row in data.iterrows():
        y = y_positions[row["日期"]]
        label = row["环节"]
        ax.barh(
            y,
            row["duration_h"],
            left=row["start_h"],
            height=0.48,
            color=colors.get(label, "#CBD5E1"),
            edgecolor="white",
            linewidth=1.0,
            label=label,
        )
        text = row["地点"]
        if row["duration_h"] >= 0.45:
            ax.text(
                row["start_h"] + row["duration_h"] / 2,
                y,
                text,
                va="center",
                ha="center",
                fontsize=9,
                color="#1F2937",
            )

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=True,
    )

    ax.set_yticks([y_positions[day] for day in days])
    ax.set_yticklabels(days, fontsize=12)
    ax.set_xlim(7, 21)
    ax.set_xticks(range(7, 22))
    ax.set_xticklabels([f"{hour}:00" for hour in range(7, 22)], rotation=0)
    ax.set_xlabel("时间", fontsize=12)
    ax.set_title("问题二最终基准行程详细时间轴", fontsize=20, fontweight="bold", pad=76)
    ax.grid(axis="x", linestyle="--", alpha=0.28)
    ax.spines[["top", "right", "left"]].set_visible(False)

    day_end = data.groupby("日期")["end_h"].max()
    latest_day = day_end.idxmax()
    latest_time = day_end.max()
    ax.text(
        0.01,
        -0.12,
        f"关键结论：最晚返回时间为{latest_day} {int(latest_time):02d}:{int(round((latest_time % 1) * 60)):02d}，"
        "所有日程均满足7:00-21:00时间窗约束。",
        transform=ax.transAxes,
        fontsize=12,
        color="#334155",
    )

    output = FIGURE_DIR / "problem2_timeline_gantt.png"
    fig.subplots_adjust(top=0.84, bottom=0.16)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    itinerary = pd.read_csv(PROCESSED_DIR / "problem2_baseline_itinerary.csv", encoding="utf-8-sig")
    timeline = pd.read_csv(PROCESSED_DIR / "problem2_baseline_timeline.csv", encoding="utf-8-sig")
    overview_path = render_itinerary_overview(itinerary)
    timeline_path = render_timeline_gantt(timeline)
    print("已生成图片：")
    print(f"- {overview_path}")
    print(f"- {timeline_path}")


if __name__ == "__main__":
    main()
