# 山东省数学建模 2026 C 题代码说明

本项目用于完成“五一”五日自驾景点优选与行程规划问题的代码求解、结果表格生成和可视化输出。代码按问题一、问题二、问题三拆分，数据统一存放在 `data/`，图表和汇总结果统一存放在 `results/`。

## 目录结构

```text
code/                  Python 源代码
data/raw/              原始景点与车程数据
data/processed/        数据处理、模型求解和验证结果
results/figures/       可视化图片
results/tables/        汇总表格与报告
```

## 环境依赖

建议使用 Python 3.10 及以上版本。安装依赖：

```bash
pip install numpy pandas matplotlib openpyxl
```

在 Windows 终端中建议使用 `-X utf8` 运行脚本，避免中文路径和中文表头出现编码问题。

## 运行顺序

1. 问题一数据处理：

```bash
python -X utf8 code/data_processing_problem1.py
```

2. 问题一模型求解：

```bash
python -X utf8 code/solve_problem1_models.py
```

3. 问题二基准行程求解与验证：

```bash
python -X utf8 code/solve_problem2_baseline_itinerary.py
```

4. 问题二行程图片生成：

```bash
python -X utf8 code/render_problem2_schedule_images.py
```

5. 问题三可靠性模拟与稳健改进分析：

```bash
python -X utf8 code/solve_problem3_reliability.py
```

## 主要输出

问题一输出：

- `data/processed/problem1_latest_topsis_result.csv`
- `data/processed/problem1_latest_type_profile.csv`
- `data/processed/problem1_latest_strong_links.csv`
- `results/figures/problem1_latest_model_visualization.png`

问题二输出：

- `data/processed/problem2_baseline_itinerary.csv`
- `data/processed/problem2_baseline_timeline.csv`
- `data/processed/problem2_pareto_solutions.csv`
- `data/processed/problem2_validation_report.csv`
- `results/figures/problem2_route_planning_diagram.png`
- `results/figures/problem2_convergence_analysis.png`
- `results/figures/problem2_weight_sensitivity_validation.png`
- `results/figures/problem2_multi_run_stability_validation.png`

问题三输出：

- `data/processed/problem3_simulation_summary.csv`
- `data/processed/problem3_disturbance_contribution.csv`
- `data/processed/problem3_reliability_scenario_grid.csv`
- `data/processed/problem3_robust_improvement_comparison.csv`
- `results/figures/problem3_reliability_dashboard.png`
- `results/figures/problem3_reliability_scenario_heatmap.png`
- `results/figures/problem3_robust_improvement_comparison.png`

## 结果口径说明

- 问题二的 P1 基准行程作为正式行程方案锁定。
- 问题二新增的权重敏感性分析和多次运行稳定性验证只作为旁路验证，不反向修改 P1。
- 问题三的稳健改进方案只用于说明如何提高可靠度，不覆盖问题二生成的正式基准行程。

## 维护说明

- 新生成的中间结果优先放入 `data/processed/`。
- 新生成的图片优先放入 `results/figures/`。
- 不建议提交 `__pycache__/`、临时提取文本、重复图片副本等文件。
