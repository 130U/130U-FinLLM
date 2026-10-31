# 24M 公开 Trial：任务与训练技术规范

本页说明公开 24M Trial 的任务、实现和计量规则。2.4B 自用版本的定位与规格见[项目介绍](PROJECT_OVERVIEW.md)和[模型版本](MODEL_CARD.md)。公开代码与本页实际运行记录对应 Trial，不以修改参数配置替代跨架构实现。

## 任务定义

输入为英文文本及指定目标，识别原文表达的投资立场：bearish、neutral、bullish。标题仅陈述经营改善、财务恶化或股价变化，并不自动表示作者明确看多或看空。目标关联、观点归属与证据范围是标签定义的一部分。

输出保留三类归一化概率、预测类别、置信度、方向分数与复核状态：

`score = 100 × (P(bullish) − P(bearish))`

分数接近零既可能来自 neutral 概率高，也可能来自两个方向概率相抵，因此不能舍弃概率向量。输入资格检查失败属于复核状态，不应以 neutral 标签掩盖。

Trial 实际训练使用标题字面实体目标模式，不等同于已验证证券代码或个人投资判断。全文补充、事件影响与个人偏好需要各自的任务定义和标签来源。

## 模型与 tokenizer

| 结构 | 24M Trial 配置 |
|---|---:|
| 总参数 | 24,255,744 |
| Transformer 层数 | 6 |
| Hidden size | 512 |
| FFN intermediate size | 1,664 |
| Query heads / KV heads | 8 / 4 |
| Head dimension | 64 |
| 词表 | 8,192 |
| 上下文长度 | 512 |
| Embedding / 输出投影 | 共享权重 |

主干为随机初始化的 dense autoregressive decoder，使用 GQA、RoPE、pre-RMSNorm、QK normalization 和 SwiGLU。tokenizer 为自行训练的 byte-level BPE。实现与配置见[模型代码](../src/finllm/model.py)、[tokenizer](../src/finllm/tokenizer.py)、[model_24m.json](../configs/model_24m.json)。

架构参考与改写来源保留在 [THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES.md)，其中 MiniMind 对照版本为 [f659b55761b754d306bd140573493a6543cafd7f](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f)。许可证与来源归属独立于版本展示，不随项目规模描述而改变。

## 阶段与实际状态

| 阶段 | Trial 状态 | 核心计量 |
|---|---|---|
| Tokenizer 与基础语言训练 | 已运行 | 词表身份、有效监督 tokens、验证 loss |
| 扩充预训练与领域适配 | 已运行 | 各阶段累计步数、金融集与通用保留集 |
| 目标条件化 SFT | 初版及平衡版已运行 | 独立样本、标签曝光、每类表现与 Macro-F1 |
| 校准与推理导出 | 已运行 | 温度、阈值、两份独立校准子集、权重绑定 |
| DPO | 有代码与合成链路检查 | 真实金融偏好训练未计入已完成实验 |
| 新的独立最终评价 | 尚未形成新的结果记录 | 事件独立、时间范围、完整预测流程 |

实际语言训练量为 26,115,096 有效监督 tokens。早期 100M 基础加 25M 领域 tokens 是 Trial 的历史预算；已消费量和预算分别记录。[状态](STATUS.md)、[分母账本](PROJECT_PROGRESS.md)及[实验报告](experiment-results.md)给出已运行数据。

## 标签与分区

2,000 条标题审核中，1,993 条已确认，包含 114 条排除判断；7 条未解决，三分类候选为 1,879 条。Trial 选择 1,008 条，分为 705 train、101 development、101 calibration、101 test；校准池再分为 51 条温度拟合和 50 条阈值选择。

人工确认的是标题范围标签。5 条标题与上下文 AI 判读分歧被保留，全文 AI 对照不是全文人工 gold。证券身份核验、事件归并和标签审核使用不同字段，不能由一个完成标志替代其他任务。详见[审核口径](TASK_REVIEW_NEXT.md)和[人工确认报告](../reports/human_headline_review_20261030.json)。

数据处理按规范化文本、近重复及已提供的关联标识分组，并冻结任务排除清单；现有 Trial 划分并非时间留出，不能据此宣称完成所有语义事件隔离。tokenizer 和语言训练语料也应检查评价内容排除情况，而不只检查 SFT 文件重叠。见[数据实现](../src/finllm/data.py)和[数据指南](DATA_GUIDE.md)。

外部来源标签不直接等同于本任务标签：公司情绪、事件利好利空与明确投资立场可能不同。新增任务与文本范围需要显式转换规则及人工复核。

## 监督目标与选型

语言训练对有效预测位置计算下一 token 损失。SFT 使用共享词表输出头，在答案标签位置计算完整词表交叉熵，prompt 和 padding 不作为标签监督；评价再取三个标签 token 的条件概率。两者监督分母不能混用。

初版训练池为 491 neutral、140 bullish、74 bearish；平衡版每轮将各类曝光补足到 491。初版 2 epochs、45 updates、1,410 标签监督位置，平衡版 3 epochs、139 updates、4,419 标签监督位置。采样与预算同时变化，后续方法比较需同时控制或披露这些条件。

语言阶段按验证 loss 选择模型，SFT 按开发集 Macro-F1 选择，并同时报告类别支持数、precision、recall 和混淆结构。开发集参与选型；已查看的旧测试集不能再包装为新的独立留出结果。相关实现见[训练器](../src/finllm/training.py)与[评价](../src/finllm/evaluation.py)。

## 状态与概率绑定

同一运行的恢复承接完整训练状态；新阶段初始化只承接相兼容权重并建立新的优化状态。训练计数区分输入位置、监督位置和 attempted/successful updates；AMP 跳过更新与成功优化步骤分别记录。梯度累积按实际有效监督位置归一化。跨架构迁移需要重建参数、词表与资源实现。

校准使用温度池拟合概率尺度，阈值池选择复核阈值，结果与 checkpoint、tokenizer 及 target mode 绑定。Trial 已运行温度约 1.50634、阈值约 0.57380；这些数值不能直接作为 2.4B 默认配置。温度缩放不改变类别 argmax，校准输出也不能替代新分布下的风险与覆盖率评价。见[导出脚本](../scripts/kaggle_inference_export.py)。

公开工程检查覆盖标签 token、因果掩码、监督位置、有限数值、检查点完整性与恢复路径；检查记录与任务结果分开保存。接续入口见[工程交接](HANDOFF.md)。

## 数据与参考来源

代码许可不自动授予训练数据再发布权。数据来源与使用条件保留在[数据指南](DATA_GUIDE.md)、[种子语料来源](research/corpus_sources_20261009.md)、[扩充语料来源](research/expanded_corpus_sources_20261011.md)和[立场任务来源](research/stance_sources_20261009.md)。

方法参考：[Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/quicktour)、[DPO 原论文](https://arxiv.org/abs/2305.18290)、[温度缩放与神经网络校准](https://proceedings.mlr.press/v70/guo17a.html)。引用方法不等于该方法在本 Trial 上已完成对应真实训练。
