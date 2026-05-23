"""Generate Problem 3 preprocessing tables and outlier boxplots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURE = ROOT / "results" / "figures"


def setup_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def parse_hour(value: str) -> float:
    hour, minute = str(value).split(":")
    return int(hour) + int(minute) / 60


def road_period(start_h: float) -> str:
    high_periods = [(7, 9), (11, 13), (16, 18)]
    return "高峰" if any(start <= start_h < end for start, end in high_periods) else "平峰"


def entry_period(start_h: float) -> str:
    return "高峰入园" if 9 <= start_h < 12 else "平峰入园"


def iqr_row(name: str, series: pd.Series) -> list:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return [name, round(q1, 4), round(q3, 4), round(iqr, 4), round(lower, 4), round(upper, 4), count, "保留" if count else "无异常"]


def main() -> None:
    setup_plot()
    FIGURE.mkdir(parents=True, exist_ok=True)

    itinerary_candidates = sorted(
        PROCESSED.glob("problem2_baseline_itinerary*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not itinerary_candidates:
        raise FileNotFoundError("未找到问题二基准行程表，请先运行问题二求解代码。")

    timeline = pd.read_csv(PROCESSED / "problem2_baseline_timeline.csv", encoding="utf-8-sig")
    attractions = pd.read_csv(PROCESSED / "attractions_processed.csv", encoding="utf-8-sig")
    commute = pd.read_csv(PROCESSED / "attraction_commute_minutes.csv", index_col=0, encoding="utf-8-sig")
    hotel = pd.read_csv(PROCESSED / "hotel_commute_minutes.csv", index_col=0, encoding="utf-8-sig")

    rows = []
    for _, record in timeline.iterrows():
        start_h = parse_hour(record["开始时间"])
        row = record.to_dict()
        row["开始小时"] = start_h
        row["结束小时"] = parse_hour(record["结束时间"])
        row["是否通勤环节"] = "是" if ("通勤" in record["环节"] or "返程" in record["环节"]) else "否"
        row["是否入园环节"] = "是" if record["环节"] == "景点游览" else "否"

        if row["是否通勤环节"] == "是":
            period = road_period(start_h)
            row["道路时段"] = period
            row["堵车延时分布"] = "U(1,4)小时" if period == "高峰" else "U(0,1.5)小时"
        else:
            row["道路时段"] = "不适用"
            row["堵车延时分布"] = "0"

        if row["是否入园环节"] == "是":
            period = entry_period(start_h)
            row["入园时段"] = period
            row["排队时长分布"] = "U(0.5,3)小时" if period == "高峰入园" else "U(0,1)小时"
        else:
            row["入园时段"] = "不适用"
            row["排队时长分布"] = "0"
        rows.append(row)

    pd.DataFrame(rows).to_csv(PROCESSED / "problem3_preprocessed_timeline_segments.csv", index=False, encoding="utf-8-sig")

    random_rules = pd.DataFrame(
        [
            ["道路堵车", "07:00-09:00、11:00-13:00、16:00-18:00", "高峰", "U(1,4)小时", "题目给定"],
            ["道路堵车", "09:00-11:00、13:00-16:00、18:00以后", "平峰", "U(0,1.5)小时", "题目给定"],
            ["入园排队", "09:00-12:00", "高峰入园", "U(0.5,3)小时", "题目给定"],
            ["入园排队", "09:00前或12:00后", "平峰入园", "U(0,1)小时", "题目给定"],
            ["固定耗时", "起床+早餐+整装", "固定流程", "1.5小时，可±20%微调", "题目给定"],
            ["固定耗时", "正餐", "固定流程", "1.0小时，可±20%微调", "题目给定"],
            ["固定耗时", "入住/退房", "固定流程", "0.5小时，可±20%微调", "题目给定"],
        ],
        columns=["数据类别", "适用时段或环节", "状态", "参数或分布", "来源说明"],
    )
    random_rules.to_csv(PROCESSED / "problem3_random_rules.csv", index=False, encoding="utf-8-sig")

    hotel_values = pd.Series(hotel.loc["酒店"].astype(float).values)
    commute_values = pd.Series(commute.replace(0, np.nan).stack().astype(float).values)
    outlier = pd.DataFrame(
        [
            iqr_row("酒店至景点车程min", hotel_values),
            iqr_row("景点间车程min", commute_values),
            iqr_row("最低必要游览时长h", attractions["min_time"].astype(float)),
            iqr_row("舒适游览时长h", attractions["comfort_time"].astype(float)),
            iqr_row("基准环节耗时h", timeline["耗时h"].astype(float)),
        ],
        columns=["变量", "Q1", "Q3", "IQR", "下界", "上界", "异常值数量", "处理方式"],
    )
    outlier.to_csv(PROCESSED / "problem3_outlier_detection.csv", index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].boxplot([hotel_values, commute_values], tick_labels=["酒店-景点", "景点-景点"])
    axes[0].set_title("通勤时间箱线图")
    axes[0].set_ylabel("分钟")
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)

    axes[1].boxplot([attractions["min_time"].astype(float), attractions["comfort_time"].astype(float)], tick_labels=["最低必要", "舒适游览"])
    axes[1].set_title("游览时长箱线图")
    axes[1].set_ylabel("小时")
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)

    axes[2].boxplot(timeline["耗时h"].astype(float), tick_labels=["基准环节"])
    axes[2].set_title("基准时间轴环节耗时箱线图")
    axes[2].set_ylabel("小时")
    axes[2].grid(axis="y", linestyle="--", alpha=0.35)

    fig.suptitle("问题三输入数据异常值检测（IQR箱线图）", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIGURE / "problem3_preprocessing_outlier_boxplots.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    print("问题三预处理输出已生成")
    print(PROCESSED / "problem3_preprocessed_timeline_segments.csv")
    print(PROCESSED / "problem3_random_rules.csv")
    print(PROCESSED / "problem3_outlier_detection.csv")
    print(FIGURE / "problem3_preprocessing_outlier_boxplots.png")


if __name__ == "__main__":
    main()
