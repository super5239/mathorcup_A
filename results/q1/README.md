# Q1 v3.1 交付：基于折外预测的相对评分

脚本：`src/q1_sensory_imputation/build_relative_scores.py`。只读取已有 5 次重复 × 3 折 OOF 预测（`analysis/A_feasibility/floor_ablation/baseline_oof.npz`）与逐维评价，**未重新训练模型**。规则见 `docs/解题思路.md` 第 1.4–1.7 节。

## 文件
| 文件 | 内容 |
|---|---|
| `q1_oof_mean_standardized_250.csv` | 250 款的 5 次重复 OOF 均值（标准化尺度，连续预测） |
| `q1_frozen_anchors.json` | 26 个 numeric 属性冻结的 P5/P95 锚点（Q3 生成谱沿用） |
| `q1_numeric_relative_scores_250.csv` | numeric 相对评分 0–6 |
| `q1_train_fold_mean_reference_250.csv` | categorical 判定用的训练折均值（每款按其 5 个测试划分对应的训练折平均） |
| `q1_categorical_ticks_250.csv` | categorical 0/1 勾选（父项门控后） |
| `q1_categorical_template_display_250.csv` | 模板展示值 = 父项强度 × 勾选 |
| `q1_prediction_support_tiers.csv` | 逐维 R²、MSE skill、预测支持度档位、最终勾选率、评分分布、锚点 |
| `3_filled_v31.xlsx` | 已填写的 12 张评分卡 + summary（原始 `3.xlsx` 未改动） |
| `q1_scorecards_12_values.csv` | 12 款评分卡对应数值，便于不经 Excel 核对 |
| `summary.json` | 汇总 |

## 结果摘要
- 预测支持度：**A 15 / B 21 / C 14**（numeric 11/8/7，categorical 4/13/7）。
- MSE skill ≤ 0（未优于风格均值）：`alcohol`、`body`。
- 无“无区分信息”属性（所有 P5 < P95）。
- numeric 相对评分分布：0 分 18.5%、1–5 分各 12–16%、6 分 9.6%。
- categorical 父项门控后最终勾选率：29.2%–58.4%，中位 44.2%。

## 评分卡填写说明
- 按每张表 `S3` 单元格的 Beer# 与 `beer` 编号匹配（beer_1–beer_12 ↔ beer 1–12），不取数据前 12 行。
- 强度在对应 0–6 格中标记 `X`，由模板原公式取分；子项勾选在勾选格标记 `X`，展示值由模板公式 = 父项强度 × 勾选。
- 每张卡 `S95` 注明“相对评分（基于折外预测，非原始品评分）”并列出 C 档属性。
- **Acetaldehyde 行（第 45 行）留空**：4.csv 无对应目标。
- **模板错误修正**：summary 表 `AX` 列（barnyard）原公式引用 `$AG$76`（Stale hops），已改为 `$AG$90`（Barnyard/horsey）。修正后经 LibreOffice 重算，summary 与脚本计算值逐格一致（最大差 0）。
- 文件由 openpyxl 写出，公式缓存值需在 Excel 打开时重算。
