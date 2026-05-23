"""
Problem 1 model solving code.

This file replaces the previous model-solving part with the latest modeling
code supplied by the user. The data-processing script is intentionally not
modified.

The model uses embedded Problem 1 data and solves:
1. indicator quantification and normalization;
2. AHP subjective weighting;
3. entropy objective weighting;
4. combined-weight TOPSIS evaluation;
5. FCM fuzzy clustering;
6. attraction linkage-combination analysis;
7. sensitivity and clustering-validity checks.
"""

from __future__ import annotations

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ModuleNotFoundError:
    plt = None
    HAS_MATPLOTLIB = False

warnings.filterwarnings("ignore")


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"


def ensure_dirs() -> None:
    """Create output folders used by this modeling script."""

    for path in [PROCESSED_DIR, TABLE_DIR, FIGURE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    """Configure Chinese fonts for matplotlib."""

    if not HAS_MATPLOTLIB:
        return
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False


def minmax_positive(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a positive indicator."""

    values = values.astype(float)
    span = values.max() - values.min()
    if math.isclose(float(span), 0.0):
        return np.ones_like(values, dtype=float)
    return (values - values.min()) / span


def minmax_negative(values: np.ndarray) -> np.ndarray:
    """Min-max normalize a negative indicator into larger-is-better form."""

    values = values.astype(float)
    span = values.max() - values.min()
    if math.isclose(float(span), 0.0):
        return np.ones_like(values, dtype=float)
    return (values.max() - values) / span


def pairwise_distance(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix, replacing scipy.spatial.distance.cdist."""

    return np.sqrt(((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2))


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average-rank implementation for Spearman correlation."""

    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and math.isclose(sorted_values[j + 1], sorted_values[i]):
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation, replacing scipy.stats.spearmanr."""

    ra = rankdata(a)
    rb = rankdata(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def input_embedded_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Step 1: input embedded raw data."""

    attractions_data = {
        "id": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
        "name": ["古城老街", "海洋乐园", "滨海浴场", "森林公园", "民俗古村", "山野溪谷", "环湖湿地", "亲子农庄", "山地观景台", "文创小镇"],
        "type": ["人文古迹", "主题游乐", "休闲度假", "自然山林", "人文乡村", "自然徒步", "生态休闲", "亲子休闲", "自然观景", "人文休闲"],
        "open_start": [8, 9, 0, 8, 8, 8, 0, 9, 8, 9],
        "open_end": [17.5, 18, 24, 17, 17.5, 17, 24, 18, 17.5, 20],
        "min_time": [2.0, 3.0, 1.0, 3.5, 2.0, 3.0, 1.5, 2.0, 1.5, 2.0],
        "comfort_time": [3.5, 5.0, 3.0, 4.5, 3.0, 4.0, 2.5, 3.0, 2.5, 3.0],
        "preference": [8.6, 9.2, 7.5, 8.0, 7.2, 7.8, 6.8, 8.3, 7.0, 7.6],
    }
    attractions = pd.DataFrame(attractions_data)

    hotel_to_att_raw = np.array([0.5, 0.8, 0.3, 1.5, 0.6, 1.2, 0.4, 0.7, 1.0, 0.5], dtype=float)
    travel_time = np.array(
        [
            [0.0, 0.4, 0.6, 1.2, 0.3, 1.0, 0.5, 0.3, 0.8, 0.2],
            [0.4, 0.0, 0.5, 1.0, 0.6, 0.9, 0.7, 0.2, 0.6, 0.3],
            [0.6, 0.5, 0.0, 1.3, 0.8, 1.4, 0.2, 0.6, 1.1, 0.5],
            [1.2, 1.0, 1.3, 0.0, 1.0, 0.4, 1.4, 0.9, 0.3, 1.1],
            [0.3, 0.6, 0.8, 1.0, 0.0, 0.8, 0.7, 0.5, 0.7, 0.4],
            [1.0, 0.9, 1.4, 0.4, 0.8, 0.0, 1.3, 0.8, 0.5, 0.9],
            [0.5, 0.7, 0.2, 1.4, 0.7, 1.3, 0.0, 0.6, 1.0, 0.4],
            [0.3, 0.2, 0.6, 0.9, 0.5, 0.8, 0.6, 0.0, 0.5, 0.3],
            [0.8, 0.6, 1.1, 0.3, 0.7, 0.5, 1.0, 0.5, 0.0, 0.7],
            [0.2, 0.3, 0.5, 1.1, 0.4, 0.9, 0.4, 0.3, 0.7, 0.0],
        ],
        dtype=float,
    )

    print("=" * 60)
    print("步骤1：数据输入完成（内嵌数据）")
    print(f"景点数量：{len(attractions)}")
    print("=" * 60)
    return attractions, hotel_to_att_raw, travel_time


def congestion_sensitivity(row: pd.Series) -> float:
    """Calculate overlap between the core entry window and peak traffic windows."""

    core_start = 8.0 if row["open_start"] == 0 else float(row["open_start"])
    core_end = core_start + 1.0
    peaks = [(7, 9), (11, 13), (16, 18)]
    overlap = 0.0
    for peak_start, peak_end in peaks:
        start = max(core_start, peak_start)
        end = min(core_end, peak_end)
        if end > start:
            overlap += end - start
    return overlap


def quantify_and_normalize(
    attractions: pd.DataFrame,
    hotel_to_att_raw: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Step 2: quantify and normalize the four model indicators."""

    attractions["congestion_sensitivity"] = attractions.apply(congestion_sensitivity, axis=1)

    ideal_duration = 3.0
    visit_duration = attractions["comfort_time"].to_numpy(dtype=float)
    max_abs_diff = np.max(np.abs(visit_duration - ideal_duration))
    attractions["visit_suitability"] = 1 - np.abs(visit_duration - ideal_duration) / max_abs_diff

    preference = attractions["preference"].to_numpy(dtype=float)
    commute_cost = hotel_to_att_raw
    sensitivity = attractions["congestion_sensitivity"].to_numpy(dtype=float)
    suitability = attractions["visit_suitability"].to_numpy(dtype=float)

    raw_data = np.column_stack([preference, suitability, commute_cost, sensitivity])
    cols = ["喜好度", "适配度", "通勤成本", "拥堵敏感度"]

    norm_data = np.zeros_like(raw_data, dtype=float)
    norm_data[:, 0] = minmax_positive(raw_data[:, 0])
    norm_data[:, 1] = minmax_positive(raw_data[:, 1])
    norm_data[:, 2] = minmax_negative(raw_data[:, 2])
    norm_data[:, 3] = minmax_negative(raw_data[:, 3])

    df_norm = pd.DataFrame(norm_data, columns=cols, index=attractions["id"])
    print("\n步骤2：指标量化与标准化完成")
    print("标准化后数据：")
    print(df_norm.round(3))
    return df_norm, norm_data, cols, preference, suitability, commute_cost, sensitivity


def ahp_weight(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """Step 3: calculate AHP subjective weight and consistency ratio."""

    n = len(matrix)
    eigvals, eigvecs = np.linalg.eig(matrix)
    max_index = int(np.argmax(eigvals.real))
    max_eig = float(eigvals[max_index].real)
    max_vec = eigvecs[:, max_index].real
    max_vec = np.abs(max_vec)
    weight = max_vec / np.sum(max_vec)

    ci = (max_eig - n) / (n - 1)
    ri_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = ri_dict[n]
    cr = ci / ri if ri != 0 else 0.0
    return weight, float(cr)


def entropy_weight(data: np.ndarray) -> np.ndarray:
    """Step 4: entropy objective weight."""

    p = data / data.sum(axis=0, keepdims=True)
    p = np.clip(p, 1e-12, 1)
    e = -1 / np.log(data.shape[0]) * np.sum(p * np.log(p), axis=0)
    d = 1 - e
    return d / np.sum(d)


def topsis(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Step 6: TOPSIS closeness coefficient."""

    weighted = data * weights
    ideal_pos = np.max(weighted, axis=0)
    ideal_neg = np.min(weighted, axis=0)
    d_pos = np.sqrt(np.sum((weighted - ideal_pos) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - ideal_neg) ** 2, axis=1))
    return d_neg / (d_pos + d_neg)


class FuzzyCMeans:
    """Simple FCM implementation matching the user's latest code structure."""

    def __init__(self, n_clusters: int = 4, m: float = 2, max_iter: int = 100, tol: float = 1e-5, random_state: int = 2026):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, x: np.ndarray) -> "FuzzyCMeans":
        rng = np.random.default_rng(self.random_state)
        n_samples, _ = x.shape
        u = rng.random((n_samples, self.n_clusters))
        u = u / np.sum(u, axis=1, keepdims=True)

        for _ in range(self.max_iter):
            um = u ** self.m
            centers = (um.T @ x) / (um.sum(axis=0)[:, np.newaxis] + 1e-12)
            dist = pairwise_distance(x, centers)
            dist = np.maximum(dist, 1e-12)
            new_u = 1.0 / np.sum((dist[:, :, np.newaxis] / dist[:, np.newaxis, :]) ** (2 / (self.m - 1)), axis=2)
            if np.linalg.norm(new_u - u) < self.tol:
                u = new_u
                break
            u = new_u

        self.U = u
        self.centers_ = centers
        self.labels_ = np.argmax(u, axis=1)
        return self


def build_topsis_result(
    attractions: pd.DataFrame,
    closeness: np.ndarray,
    preference: np.ndarray,
    suitability: np.ndarray,
    commute_cost: np.ndarray,
    sensitivity: np.ndarray,
) -> tuple[pd.DataFrame, float, float]:
    """Build the TOPSIS result table and priority levels."""

    df_topsis = pd.DataFrame(
        {
            "景点ID": attractions["id"],
            "景点名称": attractions["name"],
            "TOPSIS贴近度": closeness,
            "喜好度_raw": preference,
            "适配度_raw": suitability,
            "通勤成本_raw": commute_cost,
            "拥堵敏感度_raw": sensitivity,
        }
    ).sort_values("TOPSIS贴近度", ascending=False).reset_index(drop=True)

    q33 = float(df_topsis["TOPSIS贴近度"].quantile(0.33))
    q67 = float(df_topsis["TOPSIS贴近度"].quantile(0.67))
    df_topsis["优先级"] = df_topsis["TOPSIS贴近度"].apply(
        lambda value: "高优先级" if value >= q67 else ("中优先级" if value >= q33 else "低优先级")
    )
    return df_topsis, q33, q67


def classify_clusters(attractions: pd.DataFrame, cluster_centers: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Assign readable type-profile labels to FCM clusters."""

    cluster_labels: dict[int, str] = {}
    for cluster_id in range(len(cluster_centers)):
        center = cluster_centers.iloc[cluster_id]
        if center["喜好度"] > 0.6:
            label = "高吸引力类"
        elif center["适配度"] > 0.6:
            label = "节奏舒适类"
        elif center["通勤成本"] > 0.6:
            label = "交通便利类"
        else:
            label = "拥堵敏感类"
        cluster_labels[cluster_id] = label
    attractions["type_profile"] = attractions["cluster"].map(cluster_labels)
    return attractions


def linkage_analysis(
    attractions: pd.DataFrame,
    norm_data: np.ndarray,
    travel_time: np.ndarray,
) -> tuple[np.ndarray, list[tuple[str, str, float, float]], list[tuple[str, str, float, float]]]:
    """Step 8: calculate linkage scores and split strong/weak pairs."""

    n = len(attractions)

    def link_score(i: int, j: int) -> float:
        p_ij = (norm_data[i, 0] + norm_data[j, 0]) / 2
        t_ij = travel_time[i, j]
        max_t = np.max(travel_time[travel_time > 0])
        r_ij = 1 - t_ij / max_t
        s_ij = (norm_data[i, 3] + norm_data[j, 3]) / 2
        v_sum = attractions.iloc[i]["comfort_time"] + attractions.iloc[j]["comfort_time"]
        m_ij = max(0, 1 - abs(v_sum - 6) / 6)
        return 0.4 * p_ij + 0.35 * r_ij + 0.15 * (1 - s_ij) + 0.10 * m_ij

    link_scores = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            score = link_score(i, j)
            link_scores[i, j] = score
            link_scores[j, i] = score

    strong_links = []
    weak_links = []
    for i in range(n):
        for j in range(i + 1, n):
            t = float(travel_time[i, j])
            score = float(link_scores[i, j])
            row = (attractions.iloc[i]["id"], attractions.iloc[j]["id"], t, score)
            if t <= 0.5:
                strong_links.append(row)
            elif t <= 1.0:
                weak_links.append(row)

    strong_links_sorted = sorted(strong_links, key=lambda item: item[3], reverse=True)
    weak_links_sorted = sorted(weak_links, key=lambda item: item[3], reverse=True)
    return link_scores, strong_links_sorted, weak_links_sorted


def plot_results(
    df_topsis: pd.DataFrame,
    df_norm: pd.DataFrame,
    w_ahp: np.ndarray,
    w_entropy: np.ndarray,
    cols: list[str],
    q33: float,
    q67: float,
    link_scores: np.ndarray,
    attractions: pd.DataFrame,
) -> Path:
    """Step 9: save the combined visualization figure."""

    if not HAS_MATPLOTLIB:
        print("提示：当前 Python 环境未安装 matplotlib，已跳过可视化绘图。")
        return FIGURE_DIR / "problem1_latest_model_visualization.png"

    fig = plt.figure(figsize=(14, 10))

    ax1 = plt.subplot(2, 2, 1)
    colors_map = {"高优先级": "red", "中优先级": "orange", "低优先级": "green"}
    bar_colors = [colors_map[level] for level in df_topsis["优先级"]]
    bars = ax1.bar(df_topsis["景点ID"], df_topsis["TOPSIS贴近度"], color=bar_colors)
    ax1.axhline(y=q67, color="r", linestyle="--", label="67%分位数")
    ax1.axhline(y=q33, color="g", linestyle="--", label="33%分位数")
    ax1.set_xlabel("景点")
    ax1.set_ylabel("TOPSIS贴近度")
    ax1.set_title("景点优先级排序")
    ax1.legend()
    for bar, score in zip(bars, df_topsis["TOPSIS贴近度"]):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{score:.3f}", ha="center", fontsize=8)
    ax1.set_ylim(0, 1.1)

    ax2 = plt.subplot(2, 2, 2)
    x = np.arange(len(cols))
    width = 0.35
    ax2.bar(x - width / 2, w_ahp, width, label="AHP权重", color="skyblue")
    ax2.bar(x + width / 2, w_entropy, width, label="熵权法权重", color="lightgreen")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cols, rotation=15)
    ax2.set_ylabel("权重")
    ax2.set_title("主观与客观权重对比")
    ax2.legend()

    ax3 = plt.subplot(2, 2, 3, projection="polar")
    high_id = df_topsis[df_topsis["优先级"] == "高优先级"].iloc[0]["景点ID"]
    mid_id = df_topsis[df_topsis["优先级"] == "中优先级"].iloc[min(2, len(df_topsis[df_topsis["优先级"] == "中优先级"]) - 1)]["景点ID"]
    low_id = df_topsis[df_topsis["优先级"] == "低优先级"].iloc[-1]["景点ID"]
    angles = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
    angles += angles[:1]
    for attraction_id in [high_id, mid_id, low_id]:
        values = df_norm.loc[attraction_id].values.tolist()
        values += values[:1]
        ax3.plot(angles, values, "o-", linewidth=2, label=attraction_id)
        ax3.fill(angles, values, alpha=0.1)
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(cols, size=10)
    ax3.set_yticks([0.2, 0.5, 0.8])
    ax3.set_yticklabels(["0.2", "0.5", "0.8"], size=8)
    ax3.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))
    ax3.set_title("不同优先级景点指标雷达图")

    ax4 = plt.subplot(2, 2, 4)
    mask = np.triu(np.ones_like(link_scores), k=1).astype(bool)
    link_display = np.where(mask, link_scores, np.nan)
    im = ax4.imshow(link_display, cmap="YlOrRd")
    fig.colorbar(im, ax=ax4, label="联动得分")
    ax4.set_xticks(np.arange(len(attractions)))
    ax4.set_yticks(np.arange(len(attractions)))
    ax4.set_xticklabels(attractions["id"])
    ax4.set_yticklabels(attractions["id"])
    ax4.set_title("景点联动得分热力图（上三角）")

    fig.tight_layout()
    out_path = FIGURE_DIR / "problem1_latest_model_visualization.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_path


def sensitivity_and_validity(
    norm_data: np.ndarray,
    w_comb: np.ndarray,
    df_topsis: pd.DataFrame,
    fcm: FuzzyCMeans,
    cols: list[str],
) -> tuple[pd.DataFrame, float, pd.DataFrame]:
    """Step 10: weight-sensitivity and cluster-validity analysis."""

    print("\n步骤10：权重敏感性分析")
    delta_range = np.linspace(-0.2, 0.2, 5)
    base_closeness = topsis(norm_data, w_comb)
    # Negative closeness converts larger-is-better scores into smaller rank numbers.
    original_rank = rankdata(-base_closeness)
    rows = []
    for indicator_index, indicator_name in enumerate(cols):
        for delta in delta_range:
            w_perturb = w_comb.copy()
            w_perturb[indicator_index] = max(w_perturb[indicator_index] * (1 + delta), 1e-12)
            w_perturb = w_perturb / np.sum(w_perturb)
            new_closeness = topsis(norm_data, w_perturb)
            new_rank = rankdata(-new_closeness)
            rho = spearman_corr(original_rank, new_rank)
            max_rank_change = float(np.max(np.abs(new_rank - original_rank)))
            rows.append(
                {
                    "扰动指标": indicator_name,
                    "扰动幅度": delta,
                    "Spearman秩相关系数": rho,
                    "最大名次变化": max_rank_change,
                }
            )

    sensitivity_df = pd.DataFrame(rows)
    print("逐指标权重扰动 ±20% 时的Spearman秩相关系数：")
    summary = sensitivity_df.groupby("扰动指标").agg(
        最小秩相关系数=("Spearman秩相关系数", "min"),
        最大名次变化=("最大名次变化", "max"),
    )
    print(summary.round(4))
    if sensitivity_df["Spearman秩相关系数"].min() > 0.95:
        print("所有ρ > 0.95，排名高度稳定。")
    else:
        print("存在ρ ≤ 0.95 的扰动情形，需关注排序稳定性。")

    eta = cluster_validity(norm_data, fcm.U, fcm.centers_)
    validity_df = cluster_validity_scan(norm_data)
    print(f"\n当前K=4聚类有效性指标 η = {eta:.3f}（越大越好）")
    print("不同聚类数K的有效性比较：")
    print(validity_df.round(4).to_string(index=False))
    return sensitivity_df, eta, validity_df


def cluster_validity(x: np.ndarray, u: np.ndarray, centers: np.ndarray, m: float = 2) -> float:
    """Calculate eta = between-cluster separation / within-cluster compactness."""

    within = 0.0
    for i in range(x.shape[0]):
        for k in range(centers.shape[0]):
            within += (u[i, k] ** m) * np.sum((x[i] - centers[k]) ** 2)

    between_values = []
    for r in range(centers.shape[0]):
        for s in range(r + 1, centers.shape[0]):
            between_values.append(np.sum((centers[r] - centers[s]) ** 2))
    mean_between = float(np.mean(between_values)) if between_values else 0.0
    mean_within = within / x.shape[0]
    return float(mean_between / mean_within) if mean_within > 0 else 0.0


def cluster_validity_scan(x: np.ndarray, k_values: range = range(2, 6)) -> pd.DataFrame:
    """Compare FCM validity under different cluster counts."""

    rows = []
    for k in k_values:
        fcm_k = FuzzyCMeans(n_clusters=k, m=2, random_state=2026)
        fcm_k.fit(x)
        eta_k = cluster_validity(x, fcm_k.U, fcm_k.centers_)
        cluster_sizes = np.bincount(fcm_k.labels_, minlength=k)
        rows.append(
            {
                "聚类数K": k,
                "聚类有效性指标eta": eta_k,
                "最小类样本数": int(cluster_sizes.min()),
                "是否过度细分": "是" if cluster_sizes.min() < 2 else "否",
            }
        )
    return pd.DataFrame(rows)


def save_results(
    df_norm: pd.DataFrame,
    df_topsis: pd.DataFrame,
    attractions: pd.DataFrame,
    cluster_centers: pd.DataFrame,
    strong_links_sorted: list[tuple[str, str, float, float]],
    weak_links_sorted: list[tuple[str, str, float, float]],
    sensitivity_df: pd.DataFrame,
    eta: float,
    cluster_validity_df: pd.DataFrame,
) -> bool:
    """Step 11: save model outputs without touching data-processing files."""

    df_topsis.to_csv(PROCESSED_DIR / "problem1_latest_topsis_result.csv", index=False, encoding="utf-8-sig")
    attractions[["id", "name", "type", "cluster", "type_profile"]].to_csv(PROCESSED_DIR / "problem1_latest_type_profile.csv", index=False, encoding="utf-8-sig")
    cluster_centers.to_csv(PROCESSED_DIR / "problem1_latest_fcm_centers.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(strong_links_sorted, columns=["景点i", "景点j", "车程(h)", "联动得分"]).to_csv(PROCESSED_DIR / "problem1_latest_strong_links.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(weak_links_sorted, columns=["景点i", "景点j", "车程(h)", "联动得分"]).to_csv(PROCESSED_DIR / "problem1_latest_weak_links.csv", index=False, encoding="utf-8-sig")
    df_norm.to_csv(PROCESSED_DIR / "problem1_latest_standardized_features.csv", encoding="utf-8-sig")
    sensitivity_df.to_csv(PROCESSED_DIR / "problem1_latest_weight_sensitivity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"聚类有效性指标eta": eta}]).to_csv(PROCESSED_DIR / "problem1_latest_cluster_validity.csv", index=False, encoding="utf-8-sig")
    cluster_validity_df.to_csv(PROCESSED_DIR / "problem1_latest_cluster_validity_by_k.csv", index=False, encoding="utf-8-sig")

    try:
        with pd.ExcelWriter(TABLE_DIR / "problem1_latest_model_outputs.xlsx", engine="openpyxl") as writer:
            df_norm.to_excel(writer, sheet_name="standardized_features")
            df_topsis.to_excel(writer, sheet_name="topsis", index=False)
            attractions[["id", "name", "type", "cluster", "type_profile"]].to_excel(writer, sheet_name="fcm_profiles", index=False)
            cluster_centers.to_excel(writer, sheet_name="fcm_centers", index=False)
            pd.DataFrame(strong_links_sorted, columns=["景点i", "景点j", "车程(h)", "联动得分"]).to_excel(writer, sheet_name="strong_links", index=False)
            pd.DataFrame(weak_links_sorted, columns=["景点i", "景点j", "车程(h)", "联动得分"]).to_excel(writer, sheet_name="weak_links", index=False)
            sensitivity_df.to_excel(writer, sheet_name="sensitivity", index=False)
            pd.DataFrame([{"聚类有效性指标eta": eta}]).to_excel(writer, sheet_name="cluster_validity", index=False)
            cluster_validity_df.to_excel(writer, sheet_name="cluster_validity_by_k", index=False)
    except ModuleNotFoundError:
        print("提示：当前 Python 环境未安装 openpyxl，已跳过 Excel 汇总文件，仅保存 CSV 结果。")
        return False
    return True


def main() -> None:
    ensure_dirs()
    setup_plot_style()

    attractions, hotel_to_att_raw, travel_time = input_embedded_data()
    df_norm, norm_data, cols, preference, suitability, commute_cost, sensitivity = quantify_and_normalize(attractions, hotel_to_att_raw)

    judgment_matrix = np.array(
        [
            [1, 3, 4, 5],
            [1 / 3, 1, 2, 3],
            [1 / 4, 1 / 2, 1, 2],
            [1 / 5, 1 / 3, 1 / 2, 1],
        ],
        dtype=float,
    )
    w_ahp, cr_ahp = ahp_weight(judgment_matrix)
    print("\n步骤3：AHP主观权重")
    print(f"CR = {cr_ahp:.4f} ({'通过' if cr_ahp < 0.1 else '不通过'})")
    for i, col in enumerate(cols):
        print(f"{col}: {w_ahp[i]:.4f}")

    w_entropy = entropy_weight(norm_data)
    print("\n步骤4：熵权法客观权重")
    for i, col in enumerate(cols):
        print(f"{col}: {w_entropy[i]:.4f}")

    alpha = 0.5
    w_comb = alpha * w_ahp + (1 - alpha) * w_entropy
    w_comb = w_comb / np.sum(w_comb)
    print("\n步骤5：组合权重")
    for i, col in enumerate(cols):
        print(f"{col}: {w_comb[i]:.4f}")

    closeness = topsis(norm_data, w_comb)
    df_topsis, q33, q67 = build_topsis_result(attractions, closeness, preference, suitability, commute_cost, sensitivity)
    print("\n步骤6：TOPSIS综合评价结果")
    print(df_topsis[["景点ID", "景点名称", "TOPSIS贴近度", "优先级"]].to_string(index=False))

    fcm = FuzzyCMeans(n_clusters=4, m=2)
    fcm.fit(norm_data)
    attractions["cluster"] = fcm.labels_
    cluster_centers = pd.DataFrame(fcm.centers_, columns=cols)
    print("\n步骤7：FCM模糊聚类结果")
    print("聚类中心：")
    print(cluster_centers.round(3))

    attractions = classify_clusters(attractions, cluster_centers, cols)
    print("\n景点类型画像：")
    print(attractions[["id", "name", "type_profile"]])

    link_scores, strong_links_sorted, weak_links_sorted = linkage_analysis(attractions, norm_data, travel_time)
    print("\n步骤8：景点联动组合分析")
    print(f"强联动组合（车程≤0.5h）共{len(strong_links_sorted)}组，前5：")
    for link in strong_links_sorted[:5]:
        print(f"  {link[0]}-{link[1]} 车程{link[2]:.1f}h 得分{link[3]:.3f}")
    print(f"\n弱联动组合（0.5h<车程≤1h）共{len(weak_links_sorted)}组，前5：")
    for link in weak_links_sorted[:5]:
        print(f"  {link[0]}-{link[1]} 车程{link[2]:.1f}h 得分{link[3]:.3f}")

    figure_path = plot_results(df_topsis, df_norm, w_ahp, w_entropy, cols, q33, q67, link_scores, attractions)
    sensitivity_df, eta, cluster_validity_df = sensitivity_and_validity(norm_data, w_comb, df_topsis, fcm, cols)
    excel_saved = save_results(
        df_norm,
        df_topsis,
        attractions,
        cluster_centers,
        strong_links_sorted,
        weak_links_sorted,
        sensitivity_df,
        eta,
        cluster_validity_df,
    )

    print("\n步骤11：结果已保存。")
    if HAS_MATPLOTLIB:
        print(f"可视化图：{figure_path}")
    if excel_saved:
        print(f"Excel汇总：{TABLE_DIR / 'problem1_latest_model_outputs.xlsx'}")
    else:
        print(f"CSV结果目录：{PROCESSED_DIR}")


if __name__ == "__main__":
    main()
