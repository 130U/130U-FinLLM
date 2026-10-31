# 24M Trial｜首批数据决策记录

范围：24M Trial 历史工程研究，保留首批数据准备时的发现及决策；后续已完成 GPU 预训练、领域训练和字面实体 SFT，见[Trial 实验报告](../experiment-results.md)。项目整体见[总览](../../README.md)。

## Charter

目标：在现有 T4、24M、英文资产立场任务下，实际取得有来源依据的起步语料并生成可验证输入。范围／对象／研究单元：模型数据准备／有限资源首批真实材料／固定 Wikipedia 子集、FOMC 原创纪要、SEntFiN 候选任务池。知识深度 working；类型 descriptive。问题：Q1 使用条件；Q2 标签是否对应目标任务；Q3 规模与内存；Q4 实际执行状态。

## Canonical sources

1. 数据作者的固定仓库及数据卡：Wikimedia、SEntFiN。
2. Wikimedia 官方许可条款、Federal Reserve Board 官方再使用说明及 FOMC 日历。
3. 固定文件的大小、哈希、实际内容与本地处理结果。

精确 URL、版本及读取结果保存在 [corpus_sources_20261009.md](corpus_sources_20261009.md) 和 [stance_sources_20261009.md](stance_sources_20261009.md)。

## Findings ledger

| ID | 发现 | 证据 | 回答 |
|---|---|---|---|
| F1 | Wikipedia 固定快照包含英语文本及文章 URL，并给出 CC-BY-SA/GFDL 数据许可记录；保留与当前条款的版本差异 | 来源核查 F1–F3；原始行的 rights_note/source_url | Q1 |
| F2 | Board 原创材料除明确例外外可复制，第三方材料不自动纳入 | 官方 disclaimer、来源核查 F5–F6 | Q1 |
| F3 | SEntFiN 作者仓库声明 MIT，但实体情绪与本项目投资立场不同；原文不提供经过验证的证券代码映射 | 作者 LICENSE、论文及实际 CSV；任务来源核查 | Q1/Q2 |
| F4 | 实际取得 825 篇通用文本和 32 篇金融文本；拆分后编码并训练到 8192 词表 | reports/seed_data_preparation.json、源及预处理 manifest | Q3/Q4 |
| F5 | 首批准备时，200 条逐条 AI 初判有 40 条建议排除，且中性占多数；该批人工标签当时均未确认 | 私有候选目录 ai_review_manifest.json；非真值标签 | Q2 |
| F6 | CPU 真 token 前向反向通过；该次浏览器不能加载 Colab Monaco 编辑器，该研究阶段未执行新 GPU 训练 | reports/seed_model_input_check.json；原会话实际浏览器返回 | Q4 |

## Scope filter

排除付费新闻、X 抓取、未知上游训练条件的镜像、整库下载及收益回测。保留可追溯的限量语料；数据库许可不自动代表原网页许可。

## Object filter

Wikipedia 前缀与 FOMC 宏观文体不能泛化成推特分布；FOMC 保留抓取快照而非宣称不可变远程版本；SEntFiN 情绪必须另做资产立场审核。词法近重复检查不能证明没有语义重复。

## Conclusions

- 在本次数据准备验收时，首批下载、冻结排除池、词表及分片已完成，满足进入 24M GPU 短段试跑的条件。[F1–F4]
- 当时 SFT/DPO 尚缺人工确认的立场与资产映射；AI 初判不能替换这项证据。后续标签审核与 SFT 结果另行记录，不将来源情绪追溯当作项目人工真值。[F3/F5]
- 本次研究记录本地准备和 CPU 接口验证；随后 T4 阶段的实测结果见[训练记录](../T4_PILOT.md)。[F6]

## 当时的执行安排与风险

01a 将已验证的数据准备过程复现在 Colab 并备份，02a 训练约 1M tokens 后记录真实资源。人工审核200条，再依据可用规模扩充。金融验证只有1篇纪要、源选择有偏、语料分布窄，因此本轮不作金融效果结论，也不启动100M/25M长预算。

研究进度：1 来源排序完成；2 范围完成；3 描述性核查与证据表完成；4 范围筛选完成；5 对象筛选完成；6 结论、执行证据及未决项完成。终审为本团队的来源及一致性检查，不声称外部独立复现。
