# Q2 v3.1：统一预测支持度 + 解释稳定性 + 高度相关备选

脚本：`src/q2_compound_flavor/build_v31_tables.py`。只重组 `analysis/q2/` 已有 100 次 bootstrap 结果，**未重训、未重做 bootstrap**。

| 文件 | 内容 |
|---|---|
| `q2_v31_target_summary.csv` | 50 维：预测支持度（A/B/C，来自 Q1）、skill 标记、解释稳定性、Top-3、使用口径；旧组合分档仅作历史列 |
| `q2_v31_top3_detail.csv` | 150 条 Top-3 候选：重要性、入选率及 Wilson 区间、模块、高度相关备选 |
| `summary.json` | 汇总 |

规则：
- 解释稳定性：**单体稳定** = 3 个候选平均置换重要性均 > 0 且入选率均 ≥ 50%；否则**模块线索** = 非单点模块最高命中率 ≥ 70%；其余**探索性**。（即旧 A/B 规则去掉 skill 条件；skill 单列在预测支持度维度）
- 高度相关备选：同模块、2.csv |ρ| ≥ 0.8，最多 3 个，注明方向。

结果：
| 预测支持度 \ 解释稳定性 | 单体稳定 | 模块线索 | 探索性 |
|---|---:|---:|---:|
| A (15) | 3 | 8 | 4 |
| B (21) | 0 | 11 | 10 |
| C (14) | 0 | 4 | 10 |

- 单体稳定的 3 维：`alcohol`（但未优于风格均值）、`F_hops_all`、`aftertaste`。
- 150 条候选中 21 条有高度相关备选（如 4-ethylphenol ↔ 4-ethylguaiacol +0.81，ethanol ↔ kcalperc +0.85）。
- **Top-3 高度集中**：lactic acid 出现在 32/50 维，ethyl isovalerate 27 维，ethyl octanoate 22 维，cis-dihydro-β-terpineol 16 维。提示 Top-3 很可能大量反映风格（酸啤/野生菌类）共现，而非各感官维度特有的化学驱动。
