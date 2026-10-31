# 24M 公开 Trial：实现与验证状态

本页记录 24,255,744 参数 Trial 的代码和实际运行状态。130U-FinLLM 的整体定位为 **2.4B 自用模型、24M Trial 公开展示**，见[项目介绍](PROJECT_OVERVIEW.md)与[模型版本](MODEL_CARD.md)。下列训练计数和结果均属于 24M Trial。

## 已完成的训练链路

随机初始化 → seed 预训练 → 扩充语料续训 → 金融领域适配 → 标题实体立场 SFT → 平衡采样 SFT → 校准与推理导出。

| 阶段 | 已确认的运行与计量 | 证据入口 |
|---|---|---|
| Seed 预训练 | 300 updates；4,914,941 有效监督 tokens；验证 loss 5.5419 | [运行报告](../reports/colab_t4_continuation_20261011.json) |
| 扩充语料续训 | 累计 1,220 updates；19,988,195 有效监督 tokens；扩充验证 loss 3.6929 | [运行报告](../reports/colab_t4_expanded_continuation_20261030.json) |
| 金融领域适配 | 74 updates；1,211,960 有效监督 tokens；金融验证 loss 3.2375 | [运行报告](../reports/colab_t4_domain_20261030.json) |
| 初版实体立场 SFT | 705 条训练样本、2 epochs、45 updates、1,410 标签监督位置；开发集全部预测 neutral，Macro-F1 0.2863 | [实验报告](experiment-results.md) |
| 平衡采样 SFT | 同一训练池，3 epochs、139 updates、4,419 标签监督位置；开发集 Macro-F1 0.6549，正确 79/101 | [实验报告](experiment-results.md)、[执行脚本](../scripts/kaggle_balanced_sft.py) |
| 校准及推理导出 | 已运行；温度约 1.50634、复核阈值约 0.57380；生成私有轻量推理包 | [导出与校准脚本](../scripts/kaggle_inference_export.py) |

语言训练累计 **26,115,096 有效监督 tokens**。扩充阶段的前 610 步包含在累计 1,220 步内，不能重复相加。SFT 的标签监督位置、输入 tokens 和独立样本数分别计量。

初版全中性失败保留在实验报告中。平衡版在同一开发集上增加了 17 条方向类正确判断，同时减少了 14 条 neutral 正确判断。采样、训练轮数、更新预算和学习率日程一起变化，这一对照不能单独归因为采样。已查看的旧测试集结果属于初版 SFT。

## 数据与任务范围

人工审核任务共 2,000 条标题：已确认 1,993 条，其中包括 114 条 exclude；7 条未解决，三分类候选为 1,879 条。Trial 按标题字面目标条件选出 1,008 条，划分为 705 train、101 development、101 calibration、101 test。校准池进一步分为 51 条温度拟合与 50 条阈值选择。见[人工确认记录](../reports/human_headline_review_20261030.json)、[审核说明](TASK_REVIEW_NEXT.md)与[实体 SFT 入口](ENTITY_SFT_PILOT.md)。

确认对象是标题范围内的原文立场，不是证券身份、全文立场或未来价格方向。现有划分按文本及已提供的关联标识分组，不是时间留出；事件层面的语义独立性仍需另行审查。当前代码支持的目标表达范围也不应反推为历史训练集的样本数量。

## 公开实现

- [模型](../src/finllm/model.py)与[24M 配置](../configs/model_24m.json)：6 层 dense decoder、GQA、RoPE、QK normalization、SwiGLU、RMSNorm、共享词表权重。
- [数据处理](../src/finllm/data.py)：任务分组、排除清单、token 打包及内存映射读取；[tokenizer](../src/finllm/tokenizer.py)为自训 8K byte-level BPE。
- [训练器](../src/finllm/training.py)：语言训练、答案位置 SFT、DPO 路径、梯度累积、训练计数与检查点恢复。
- [校准](../src/finllm/evaluation.py)与[推理](../src/finllm/inference.py)：三类概率、方向分数、输入检查及复核状态。

DPO 有实现和小型合成链路检查，尚无本页所述真实金融数据的 DPO 运行结果。[Twitter 弱监督脚本](../scripts/kaggle_twitter_weak_sft.py)也不属于上述已确认实验。后续评价应使用新的独立样本，分别报告各类别与接受样本的表现。

## 证据与入口

初版训练产物来自已核验的私有备份；平衡 SFT 和推理导出完成状态来自用户提供的运行输出。公开仓库提供代码、配置与汇总记录，私有标题、标签、权重和备份不随仓库发布。具体结果归属见[实验报告](experiment-results.md)。

[CPU smoke](../reports/cpu_smoke_check.json)、[完整配置 CPU 检查](../reports/full_config_cpu_check.json)、[续训检查](../reports/seed_continuation_checks.json)和[领域交接检查](../reports/domain_handoff_checks_20261030.json)记录工程验证，不能替代任务能力评价。历史 reports 保留各自当时状态；其中早期“等待审核”“SFT 尚未启动”等描述已由后续实际运行补充。

继续阅读：[训练与数据分母](PROJECT_PROGRESS.md) · [工程交接入口](HANDOFF.md) · [Trial 技术规范](PLAN.md)。
