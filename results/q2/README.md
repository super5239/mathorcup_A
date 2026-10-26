# Q2 v3.1：统一预测支持度 + 解释稳定性 + 高度相关备选

脚本：`src/q2_compound_flavor/build_v31_tables.py`。只重组 `analysis/q2/` 已有 100 次 bootstrap 结果，**未重训、未重做 bootstrap**。

| 文件 | 内容 |
|---|---|
| `q2_v31_target_summary.csv` | 50 维：预测支持度（A/B/C，来自 Q1）、skill 标记、解释稳定性、Top-3、使用口径；旧组合分档仅作历史列 |
| `q2_v31_top3_detail.csv` | 150 条 Top-3 候选：重要性、入选率及 Wilson 区间、模块、高度相关备选 |
| `summary.json` | 汇总 |

规则：
- 解释稳定性：**单体稳定** = 3 个候选平均置换重要性均 > 0 且入选率均 ≥ 50%；否则**模块线索** = 非单点模块最高命中率 ≥ 70%；其余**探索性**。（即旧 A/B 规则去掉 skill 条件；skill 单列在预测支持度维度）
- 高度相关备选：同模块、2.csv |ρ| ≥ 0.8，最多 3 个，注明方向。仅列单体化合物；理化/派生/混合指标（如 `kcalperc`）单列为“派生相关指标”，不作化合物备选。

结果：
| 预测支持度 \ 解释稳定性 | 单体稳定 | 模块线索 | 探索性 |
|---|---:|---:|---:|
| A (15) | 3 | 8 | 4 |
| B (21) | 0 | 11 | 10 |
| C (14) | 0 | 4 | 10 |

- 单体稳定的 3 维：`alcohol`（但未优于风格均值）、`F_hops_all`、`aftertaste`。
- 150 条候选中 12 条有化合物高度相关备选（如 4-ethylphenol ↔ 4-ethylguaiacol +0.81）；9 条仅有派生相关指标（ethanol ↔ kcalperc +0.85）。
- **Top-3 高度集中**：lactic acid 出现在 32/50 维，ethyl isovalerate 27 维，ethyl octanoate 22 维，cis-dihydro-β-terpineol 16 维。提示 Top-3 很可能大量反映风格（酸啤/野生菌类）共现，而非各感官维度特有的化学驱动。

## 敏感性实验：全局置换 vs 风格内置换
脚本：`src/q2_compound_flavor/within_style_permutation.py`（约 40 秒）。

- 沿用 Q1 正式 OOF 的 15 个划分与 Extra Trees 固定配置，**重新拟合 15 次**；重拟合预测与保存的 OOF 逐值一致（最大差 7e-16）。不改动原 OOF、不重做 100 次 bootstrap、不重算入选率、不替换主名单。
- 同一测试折、同一模型、同一评价集合上配对比较两种置换（各 3 次重复）。评价集合 = 测试折内同风格 ≥2 款的样本：1215/1250 可评价（每折排除 2–3 款单独出现的风格样本）。
- 风格内重要性只支持“在已记录风格分组内仍存在预测依赖”，不是化学因果结论。风格内置换只打乱风格内变异，其绝对量天然小于全局置换，因此**保留比例偏低本身不等于混杂**，排序保留更有参考价值。

| 指标（150 条 Top-3 候选） | 结果 |
|---|---:|
| 全局置换重要性 > 0 | 99.3% |
| 风格内置换重要性 > 0 | 83.3% |
| 保留比例（风格内/全局）中位数 | 0.187 |
| 风格内仍进入 217 个候选的 Top-10 | 72% |
| 风格内仍进入 Top-3 | 54% |
| 目标内 3 个候选原排序完全保留 | 10/50 维 |

| 化合物 | 出现维度 | 保留比例中位 | 风格内 >0 | 风格内仍在 Top-10 |
|---|---:|---:|---:|---:|
| lactic acid | 32 | **0.02** | 59% | 44% |
| ethyl isovalerate | 27 | 0.24 | 93% | 85% |
| ethyl octanoate | 22 | 0.24 | 86% | 82% |
| cis-dihydro-β-terpineol | 16 | 0.26 | 94% | 88% |
| ethanol | 9 | 0.13 | 67% | 56% |
| ethyl acetate | 7 | 0.21 | 100% | 86% |

解读要点：
- **乳酸的普遍入选主要是风格代理**：对 bitternes、alcohol、X4vg，乳酸风格内重要性≈0（风格内排名 212、155、40）。
- 但在 **acidity、lactic、acetic** 上，乳酸风格内仍排第 1（保留比例 6%–10%，15 折中 13–14 折为正），与化学预期一致。
- **X4vg ↔ 4VG** 风格内保留 96%、排名第 1，是最干净的“化合物—感官”对应。
- 按预测支持度：A 档候选风格内仍在 Top-10 的比例 82%，B 档 73%，C 档 60%。

文件：`q2_within_style_permutation_top3.csv`（逐候选）、`q2_within_style_permutation_by_field.csv`、`q2_within_style_permutation_log.csv`（逐折可评价样本与排除风格）、`perm_global_importance_mean15.csv`、`perm_within_style_importance_mean15.csv`、`within_style_permutation_summary.json`。
