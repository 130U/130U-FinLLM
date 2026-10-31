# 24M 公开 Trial：工程交接

本页用于接续公开 24M Trial 的代码与记录。项目的 2.4B 自用版本见[项目介绍](PROJECT_OVERVIEW.md)及[模型版本](MODEL_CARD.md)，其架构不能直接替换本仓库的默认配置后视为已完成迁移。

Trial 已完成随机初始化预训练、金融适配、标题实体立场 SFT、平衡采样对照和校准推理导出。语言训练累计 26,115,096 有效监督 tokens；最终 SFT 训练池为 705 条独立样本，平衡运行共 4,419 标签监督位置。状态与结果分别见[实现状态](STATUS.md)、[分母账本](PROJECT_PROGRESS.md)和[实验报告](experiment-results.md)。

## 工程入口

| 工作 | 入口 | 使用条件 |
|---|---|---|
| 安装与合成 smoke | [项目依赖](../pyproject.toml)、[00 Notebook](../notebooks/00_setup_and_smoke.ipynb)、[CLI](../src/finllm/cli.py) | 验证代码链路，不代表真实金融训练结果 |
| Seed 数据和预训练 | [SEED_DATA](SEED_DATA.md)、[T4_PILOT](T4_PILOT.md) | 使用相匹配的语料、tokenizer 和排除清单 |
| 扩充与续训 | [DATA_EXPANSION](DATA_EXPANSION.md)、[EXPANDED_CONTINUATION](EXPANDED_CONTINUATION.md) | 区分早期中间检查点与后续累计记录 |
| 领域适配与任务衔接 | [DOMAIN_HANDOFF](DOMAIN_HANDOFF.md)、[POST_DOMAIN_NEXT](POST_DOMAIN_NEXT.md) | 核对母权重、词表、数据与阶段配置 |
| 实体立场 SFT | [ENTITY_SFT_PILOT](ENTITY_SFT_PILOT.md)、[04e Notebook](../notebooks/04e_kaggle_lite_sft.ipynb) | 需要私有任务包及匹配的领域权重 |
| 平衡采样对照 | [执行脚本](../scripts/kaggle_balanced_sft.py) | 依赖已准备的 04e 会话输入，训练分区内采样 |
| 校准及推理导出 | [执行脚本](../scripts/kaggle_inference_export.py) | 使用平衡版最佳权重、原词表及两份校准子集 |
| 人工审核和导入 | [TASK_REVIEW_NEXT](TASK_REVIEW_NEXT.md)、[REVIEW_WORKSPACE](REVIEW_WORKSPACE.md) | 核对来源版本、标签范围和人工确认状态 |

仓库提供代码、默认配置和公开汇总，不包含私有标题、标签、训练权重或完整运行备份。历史真实运行的复现需要匹配的数据、检查点与运行配置；通用配置文件不是每次实验的完整清单。

## 三类状态承接

**同一运行恢复。** 使用完整检查点承接模型、优化器、学习率日程、随机状态、数据位置与进度，并核对配置及来源身份。[训练器](../src/finllm/training.py)和[检查点模块](../src/finllm/checkpoint.py)包含兼容性与完整性检查。已完成运行的只读恢复检查不新增训练量。

**阶段切换。** 通过初始化检查点承接相兼容的模型权重，建立新阶段的优化器与数据进度。领域适配到 SFT、初版对照与平衡版从共同领域母权重开始，均应与完整 resume 区分。

**跨架构扩容。** 24M 与 2.4B 的词表、维度、层数和参数形状不同。可复用任务规则、数据身份与验证方法；token IDs、模型权重、优化器状态、温度和阈值不能作为同一运行直接搬入新架构。

轻量推理包只包含预测所需权重与元数据，不含完整 optimizer/RNG 状态，不能替代恢复检查点。完整检查点、配对元数据、完成标志及数据版本应共同保留；跨硬件恢复不等于逐位数值一致。

## 接续实验应记录的最小账本

每次新实验使用独立输出目录，记录代码 revision、模型配置、tokenizer 身份、母检查点、数据及分区版本、目标语义、采样规则、输入/监督计数、尝试/成功更新、选型集与校准集。比较采样策略时，同时核对标签曝光、更新数与学习率日程，避免将更多训练归为单一方法收益。

后续独立评价应重新冻结样本，避免把已经查看的旧测试集当作新结果。真实 DPO 需要真实偏好监督；标题三分类标签不是天然的偏好对。已有 DPO 实现和合成 smoke 保留为工程入口。

## 保存与发布边界

保留所有历史 reports，包括失败与早期准备状态；当前状态以 [STATUS](STATUS.md) 和[实验报告](experiment-results.md)补充。公开代码许可与第三方来源说明见 [LICENSE](../LICENSE)、[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)；数据使用按[数据指南](DATA_GUIDE.md)及各来源条件管理。
