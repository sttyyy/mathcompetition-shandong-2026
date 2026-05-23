"""
Problem C - Problem 1 data processing.

This script follows the data-processing specification in
data/problem1_data_processing_spec.txt:
1. clean and validate the attraction and commute data;
2. convert commute time to minutes and derive open duration;
3. quantify congestion sensitivity and visit-duration suitability;
4. normalize features for TOPSIS/FCM style downstream models;
5. generate linkage-pair data and lightweight visual outputs.
"""

from __future__ import annotations

import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"

ATTRACTIONS_PATH = RAW_DIR / "attractions.csv"
DISTANCE_PATH = RAW_DIR / "distance_matrix.csv"

IDEAL_VISIT_HOURS = 3.0
FULL_DAY_OPEN_START = 8.0
FULL_DAY_OPEN_END = 22.0
PEAK_WINDOWS = [(7.0, 9.0), (11.0, 13.0), (16.0, 18.0)]


def ensure_dirs() -> None:
    for path in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]
    for font_path in candidates:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


def minmax_positive(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if math.isclose(float(hi), float(lo)):
        return pd.Series(1.0, index=series.index)
    return (series - lo) / (hi - lo)


def minmax_negative(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if math.isclose(float(hi), float(lo)):
        return pd.Series(1.0, index=series.index)
    return (hi - series) / (hi - lo)


def overlap_hours(interval_a: tuple[float, float], interval_b: tuple[float, float]) -> float:
    start = max(interval_a[0], interval_b[0])
    end = min(interval_a[1], interval_b[1])
    return max(0.0, end - start)


def iqr_flags(series: pd.Series) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def validate_inputs(attractions: pd.DataFrame, distance: pd.DataFrame) -> list[str]:
    messages: list[str] = []

    if attractions["id"].duplicated().any():
        dupes = attractions.loc[attractions["id"].duplicated(), "id"].tolist()
        messages.append(f"景点编号存在重复: {dupes}")
    else:
        messages.append("景点编号唯一性检查通过。")

    missing_attr = attractions.isna().sum()
    if int(missing_attr.sum()) == 0:
        messages.append("景点基础信息未发现缺失值。")
    else:
        messages.append(f"景点基础信息缺失值统计: {missing_attr[missing_attr > 0].to_dict()}")

    if int(distance.isna().sum().sum()) == 0:
        messages.append("通勤矩阵未发现缺失值。")
    else:
        messages.append(f"通勤矩阵缺失值数量: {int(distance.isna().sum().sum())}")

    preference_ok = attractions["preference"].between(1, 10).all()
    messages.append(f"喜好度评分范围检查: {'通过' if preference_ok else '存在超出1-10范围的评分'}。")

    time_ok = (
        attractions["open_start"].between(0, 24).all()
        and attractions["open_end"].between(0, 24).all()
        and (attractions["open_end"] >= attractions["open_start"]).all()
    )
    messages.append(f"开放时间格式检查: {'通过' if time_ok else '存在异常开放时间'}。")

    symmetry_gap = float((distance - distance.T).abs().to_numpy().max())
    diagonal_gap = float(np.abs(np.diag(distance.to_numpy())).max())
    messages.append(f"通勤矩阵对称性最大偏差: {symmetry_gap:.6f}；对角线最大偏差: {diagonal_gap:.6f}。")

    for col in ["preference", "min_time", "comfort_time"]:
        flagged = attractions.loc[iqr_flags(attractions[col]), "id"].tolist()
        messages.append(f"{col} IQR异常检测: {'未发现异常值' if not flagged else flagged}。")

    distance_values = distance.to_numpy().reshape(-1)
    distance_flags = iqr_flags(pd.Series(distance_values))
    messages.append(
        f"车程数据IQR异常检测: {'未发现异常值' if not distance_flags.any() else '发现疑似异常值'}。"
    )

    return messages


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    attractions = pd.read_csv(ATTRACTIONS_PATH, encoding="utf-8")
    distance = pd.read_csv(DISTANCE_PATH, index_col=0, encoding="utf-8")
    distance = distance.astype(float)
    return attractions, distance


def derive_features(attractions: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    df = attractions.copy()
    full_day = (df["open_start"] == 0) & (df["open_end"] == 24)
    df["effective_open_start"] = np.where(full_day, FULL_DAY_OPEN_START, df["open_start"])
    df["effective_open_end"] = np.where(full_day, FULL_DAY_OPEN_END, df["open_end"])
    df["open_duration_h"] = df["effective_open_end"] - df["effective_open_start"]

    entry_windows = list(zip(df["effective_open_start"], df["effective_open_start"] + 1.0))
    df["congestion_overlap_h"] = [
        sum(overlap_hours(window, peak) for peak in PEAK_WINDOWS) for window in entry_windows
    ]
    df["congestion_sensitivity"] = df["congestion_overlap_h"] / 1.0

    max_visit_gap = (df["comfort_time"] - IDEAL_VISIT_HOURS).abs().max()
    df["visit_suitability"] = 1 - (df["comfort_time"] - IDEAL_VISIT_HOURS).abs() / max_visit_gap

    hotel_commute_h = distance.loc["酒店", df["id"]]
    df["hotel_commute_h"] = df["id"].map(hotel_commute_h.to_dict())
    df["hotel_commute_min"] = df["hotel_commute_h"] * 60

    df["type_code"], _ = pd.factorize(df["type"])
    return df


def normalized_features(features: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame({"id": features["id"], "name": features["name"], "type": features["type"]})
    normalized["preference_norm"] = minmax_positive(features["preference"])
    normalized["visit_suitability_norm"] = minmax_positive(features["visit_suitability"])
    normalized["hotel_commute_norm"] = minmax_negative(features["hotel_commute_min"])
    normalized["congestion_resilience_norm"] = minmax_negative(features["congestion_sensitivity"])
    normalized["open_duration_norm"] = minmax_positive(features["open_duration_h"])
    normalized["priority_input_score"] = normalized[
        [
            "preference_norm",
            "visit_suitability_norm",
            "hotel_commute_norm",
            "congestion_resilience_norm",
        ]
    ].mean(axis=1)
    return normalized


def build_linkage_pairs(features: pd.DataFrame, distance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    feature_by_id = features.set_index("id")
    for a, b in itertools.combinations(features["id"], 2):
        t_h = float(distance.loc[a, b])
        if t_h <= 0.5:
            level = "强联动"
        elif t_h <= 1.0:
            level = "弱联动"
        else:
            level = "不推荐"

        visit_sum = float(feature_by_id.loc[[a, b], "comfort_time"].sum())
        avg_preference = float(feature_by_id.loc[[a, b], "preference"].mean())
        avg_resilience = 1 - float(feature_by_id.loc[[a, b], "congestion_sensitivity"].mean())
        linkage_score = (
            0.40 * minmax_pair_preference(avg_preference)
            + 0.35 * (1 - min(t_h, 1.5) / 1.5)
            + 0.15 * avg_resilience
            + 0.10 * max(0.0, 1 - abs(visit_sum - 6.0) / 6.0)
        )
        rows.append(
            {
                "spot_i": a,
                "spot_j": b,
                "name_i": feature_by_id.loc[a, "name"],
                "name_j": feature_by_id.loc[b, "name"],
                "commute_h": t_h,
                "commute_min": t_h * 60,
                "linkage_level": level,
                "comfort_time_sum_h": visit_sum,
                "avg_preference": avg_preference,
                "avg_congestion_resilience": avg_resilience,
                "linkage_score": linkage_score,
            }
        )
    pairs = pd.DataFrame(rows).sort_values(["linkage_level", "linkage_score"], ascending=[True, False])
    level_order = pd.CategoricalDtype(["强联动", "弱联动", "不推荐"], ordered=True)
    pairs["linkage_level"] = pairs["linkage_level"].astype(level_order)
    return pairs.sort_values(["linkage_level", "linkage_score"], ascending=[True, False]).reset_index(drop=True)


def minmax_pair_preference(value: float) -> float:
    return (value - 6.8) / (9.2 - 6.8)


def fuzzy_c_means(data: np.ndarray, cluster_count: int = 3, m: float = 2.0, max_iter: int = 200) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2026)
    n = data.shape[0]
    membership = rng.random((n, cluster_count))
    membership = membership / membership.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        old = membership.copy()
        um = membership**m
        centers = (um.T @ data) / um.sum(axis=0)[:, None]
        dist = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        dist = np.maximum(dist, 1e-8)
        inv = dist ** (-2 / (m - 1))
        membership = inv / inv.sum(axis=1, keepdims=True)
        if np.max(np.abs(membership - old)) < 1e-6:
            break
    labels = membership.argmax(axis=1)
    return labels, centers


def add_fcm_clusters(features: pd.DataFrame, normalized: pd.DataFrame) -> pd.DataFrame:
    cluster_input = normalized[
        ["preference_norm", "visit_suitability_norm", "hotel_commute_norm", "congestion_resilience_norm"]
    ].to_numpy()
    labels, _ = fuzzy_c_means(cluster_input, cluster_count=3)
    out = normalized[["id", "name", "type", "priority_input_score"]].copy()
    out["cluster_id"] = labels
    cluster_mean = out.groupby("cluster_id")["priority_input_score"].mean().sort_values(ascending=False)
    cluster_name_map = {
        cluster_id: name
        for cluster_id, name in zip(cluster_mean.index, ["高优先级", "中优先级", "低优先级"])
    }
    out["priority_level"] = out["cluster_id"].map(cluster_name_map)
    return features.merge(out[["id", "cluster_id", "priority_level"]], on="id", how="left")


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font, fill=(30, 30, 30), anchor=None) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def value_to_color(value: float) -> tuple[int, int, int]:
    value = float(np.clip(value, 0, 1))
    low = np.array([238, 244, 250])
    high = np.array([32, 112, 162])
    return tuple((low + (high - low) * value).astype(int))


def draw_bar_chart(df: pd.DataFrame, value_col: str, title: str, path: Path) -> None:
    data = df.sort_values(value_col, ascending=True).reset_index(drop=True)
    width, height = 1200, 760
    margin_l, margin_r, margin_t, margin_b = 170, 80, 90, 70
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = get_font(32, True), get_font(20), get_font(18)
    draw_text(draw, (width // 2, 38), title, title_font, anchor="mm")
    max_v = float(data[value_col].max())
    min_v = float(data[value_col].min())
    span = max(max_v - min_v, 1e-9)
    bar_h = (height - margin_t - margin_b) / len(data)
    for i, row in data.iterrows():
        y = margin_t + i * bar_h + 8
        v = float(row[value_col])
        bar_w = int((width - margin_l - margin_r) * ((v - min_v) / span if span else 1))
        color = value_to_color((v - min_v) / span)
        draw_text(draw, (20, int(y + bar_h * 0.35)), f"{row['id']} {row['name']}", label_font)
        draw.rounded_rectangle([margin_l, y, margin_l + bar_w, y + bar_h - 10], radius=6, fill=color)
        draw_text(draw, (margin_l + bar_w + 10, int(y + bar_h * 0.35)), f"{v:.3f}", small_font)
    draw.line([margin_l, height - margin_b, width - margin_r, height - margin_b], fill=(150, 150, 150), width=2)
    img.save(path)


def draw_heatmap(matrix: pd.DataFrame, title: str, path: Path, fmt: str = ".0f") -> None:
    labels = list(matrix.index)
    width, height = 1050, 930
    margin_l, margin_t = 170, 130
    cell = min((width - margin_l - 60) // len(labels), (height - margin_t - 70) // len(labels))
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = get_font(32, True), get_font(18), get_font(15)
    draw_text(draw, (width // 2, 45), title, title_font, anchor="mm")

    values = matrix.to_numpy(dtype=float)
    lo, hi = values.min(), values.max()
    span = max(hi - lo, 1e-9)
    for j, label in enumerate(labels):
        draw_text(draw, (margin_l + j * cell + cell // 2, margin_t - 24), label, label_font, anchor="mm")
    for i, label in enumerate(labels):
        draw_text(draw, (margin_l - 18, margin_t + i * cell + cell // 2), label, label_font, anchor="rm")
        for j in range(len(labels)):
            v = float(values[i, j])
            color = value_to_color((v - lo) / span)
            x0, y0 = margin_l + j * cell, margin_t + i * cell
            draw.rectangle([x0, y0, x0 + cell - 2, y0 + cell - 2], fill=color)
            text_color = "white" if (v - lo) / span > 0.55 else (35, 35, 35)
            draw_text(draw, (x0 + cell // 2, y0 + cell // 2), format(v, fmt), small_font, fill=text_color, anchor="mm")
    img.save(path)


def draw_feature_heatmap(normalized: pd.DataFrame, path: Path) -> None:
    cols = [
        "preference_norm",
        "visit_suitability_norm",
        "hotel_commute_norm",
        "congestion_resilience_norm",
        "open_duration_norm",
    ]
    labels = ["喜好度", "时长适配", "通勤便捷", "抗拥堵", "开放时长"]
    data = normalized.sort_values("priority_input_score", ascending=False).reset_index(drop=True)
    width, height = 1100, 760
    margin_l, margin_t = 180, 120
    cell_w, cell_h = 165, 52
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = get_font(32, True), get_font(19), get_font(16)
    draw_text(draw, (width // 2, 45), "景点标准化特征矩阵", title_font, anchor="mm")
    for j, label in enumerate(labels):
        draw_text(draw, (margin_l + j * cell_w + cell_w // 2, margin_t - 28), label, label_font, anchor="mm")
    for i, row in data.iterrows():
        y = margin_t + i * cell_h
        draw_text(draw, (margin_l - 18, y + cell_h // 2), f"{row['id']} {row['name']}", label_font, anchor="rm")
        for j, col in enumerate(cols):
            v = float(row[col])
            x = margin_l + j * cell_w
            draw.rectangle([x, y, x + cell_w - 3, y + cell_h - 3], fill=value_to_color(v))
            draw_text(draw, (x + cell_w // 2, y + cell_h // 2), f"{v:.2f}", small_font, fill=("white" if v > 0.55 else (35, 35, 35)), anchor="mm")
    img.save(path)


def draw_cluster_scatter(clustered: pd.DataFrame, normalized: pd.DataFrame, path: Path) -> None:
    plot = clustered.merge(normalized[["id", "preference_norm", "hotel_commute_norm"]], on="id")
    width, height = 980, 720
    margin = 100
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = get_font(32, True), get_font(19), get_font(16)
    colors = {"高优先级": (205, 70, 49), "中优先级": (41, 128, 185), "低优先级": (120, 135, 145)}
    draw_text(draw, (width // 2, 42), "FCM景点类型画像：喜好度-通勤便捷性", title_font, anchor="mm")
    x0, y0, x1, y1 = margin, height - margin, width - margin, margin
    draw.line([x0, y0, x1, y0], fill=(80, 80, 80), width=2)
    draw.line([x0, y0, x0, y1], fill=(80, 80, 80), width=2)
    draw_text(draw, ((x0 + x1) // 2, height - 42), "通勤便捷性（标准化）", label_font, anchor="mm")
    draw_text(draw, (42, (y0 + y1) // 2), "喜好度（标准化）", label_font, anchor="mm")
    for _, row in plot.iterrows():
        x = x0 + float(row["hotel_commute_norm"]) * (x1 - x0)
        y = y0 - float(row["preference_norm"]) * (y0 - y1)
        c = colors[row["priority_level"]]
        draw.ellipse([x - 11, y - 11, x + 11, y + 11], fill=c, outline=(40, 40, 40))
        draw_text(draw, (int(x + 14), int(y - 8)), row["id"], small_font)
    legend_x = width - 240
    for i, (name, color) in enumerate(colors.items()):
        y = 110 + i * 34
        draw.rectangle([legend_x, y, legend_x + 22, y + 22], fill=color)
        draw_text(draw, (legend_x + 32, y - 1), name, label_font)
    img.save(path)


def draw_linkage_chart(pairs: pd.DataFrame, path: Path) -> None:
    strong = pairs[pairs["linkage_level"] == "强联动"].head(12)
    width, height = 1420, 760
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font, label_font, small_font = get_font(32, True), get_font(20), get_font(17)
    draw_text(draw, (width // 2, 42), "强联动候选组合（车程≤30分钟）", title_font, anchor="mm")
    max_score = float(strong["linkage_score"].max())
    x0, y = 460, 110
    for _, row in strong.iterrows():
        score = float(row["linkage_score"])
        bar_w = int((width - x0 - 185) * score / max_score)
        label = f"{row['spot_i']}-{row['spot_j']}  {row['name_i']} + {row['name_j']}"
        draw_text(draw, (30, y + 8), label, label_font)
        draw.rounded_rectangle([x0, y, x0 + bar_w, y + 30], radius=6, fill=(53, 126, 159))
        draw_text(draw, (x0 + bar_w + 12, y + 4), f"{score:.3f} / {row['commute_min']:.0f}分钟", small_font)
        y += 48
    img.save(path)


def write_outputs(
    features: pd.DataFrame,
    normalized: pd.DataFrame,
    clustered: pd.DataFrame,
    distance: pd.DataFrame,
    pairs: pd.DataFrame,
    validation_messages: list[str],
) -> None:
    distance_min = distance * 60
    hotel_commute = distance_min.loc[["酒店"], features["id"]]
    attraction_commute = distance_min.loc[features["id"], features["id"]]
    congestion_table = features[
        [
            "id",
            "name",
            "effective_open_start",
            "effective_open_end",
            "congestion_overlap_h",
            "congestion_sensitivity",
        ]
    ]
    topsis_input = normalized[
        [
            "id",
            "name",
            "type",
            "preference_norm",
            "visit_suitability_norm",
            "hotel_commute_norm",
            "congestion_resilience_norm",
        ]
    ]

    clustered.to_csv(PROCESSED_DIR / "attractions_processed.csv", index=False, encoding="utf-8-sig")
    normalized.to_csv(PROCESSED_DIR / "standardized_features.csv", index=False, encoding="utf-8-sig")
    hotel_commute.to_csv(PROCESSED_DIR / "hotel_commute_minutes.csv", encoding="utf-8-sig")
    attraction_commute.to_csv(PROCESSED_DIR / "attraction_commute_minutes.csv", encoding="utf-8-sig")
    congestion_table.to_csv(PROCESSED_DIR / "congestion_sensitivity.csv", index=False, encoding="utf-8-sig")
    pairs.to_csv(PROCESSED_DIR / "linkage_pairs.csv", index=False, encoding="utf-8-sig")
    topsis_input.to_csv(PROCESSED_DIR / "topsis_input.csv", index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(TABLE_DIR / "problem1_data_processing_outputs.xlsx", engine="openpyxl") as writer:
        clustered.to_excel(writer, sheet_name="processed_attractions", index=False)
        normalized.to_excel(writer, sheet_name="standardized_features", index=False)
        hotel_commute.to_excel(writer, sheet_name="hotel_commute_min")
        attraction_commute.to_excel(writer, sheet_name="attraction_commute_min")
        congestion_table.to_excel(writer, sheet_name="congestion_sensitivity", index=False)
        pairs.to_excel(writer, sheet_name="linkage_pairs", index=False)
        topsis_input.to_excel(writer, sheet_name="topsis_input", index=False)

    report_lines = [
        "问题一数据处理质量检查报告",
        "=" * 32,
        *validation_messages,
        "",
        f"输出景点数量: {len(features)}",
        f"强联动组合数量: {int((pairs['linkage_level'] == '强联动').sum())}",
        f"弱联动组合数量: {int((pairs['linkage_level'] == '弱联动').sum())}",
        f"不推荐联动组合数量: {int((pairs['linkage_level'] == '不推荐').sum())}",
    ]
    (TABLE_DIR / "problem1_data_processing_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    draw_bar_chart(normalized, "priority_input_score", "景点数据处理综合输入得分", FIGURE_DIR / "problem1_priority_input_score.png")
    draw_feature_heatmap(normalized, FIGURE_DIR / "problem1_standardized_feature_heatmap.png")
    draw_heatmap(attraction_commute, "景点间基准车程矩阵（分钟）", FIGURE_DIR / "problem1_commute_matrix_heatmap.png", fmt=".0f")
    draw_cluster_scatter(clustered, normalized, FIGURE_DIR / "problem1_fcm_cluster_scatter.png")
    draw_linkage_chart(pairs, FIGURE_DIR / "problem1_strong_linkage_pairs.png")


def main() -> None:
    ensure_dirs()
    attractions, distance = load_data()
    validation_messages = validate_inputs(attractions, distance)
    features = derive_features(attractions, distance)
    normalized = normalized_features(features)
    clustered = add_fcm_clusters(features, normalized)
    pairs = build_linkage_pairs(features, distance)
    write_outputs(clustered, normalized, clustered, distance, pairs, validation_messages)

    print("问题一数据处理完成。")
    print(f"处理后数据目录: {PROCESSED_DIR}")
    print(f"表格汇总: {TABLE_DIR / 'problem1_data_processing_outputs.xlsx'}")
    print(f"可视化目录: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
