# 扩充语料与新阶段权重承接

2026-10-11建立，2026-10-30更新。本轮入口承接已完成的 `seed_v1` 300 步成果。原数据、8192-token 词表、检查点与报告归档均保留；新产物单独放在 `expansion_v1`。用户已完成01b、02c和02d，扩充累计1220步、19,988,195个有效tokens，最终快照输出校验通过。最新证据见[1220步读回记录](../reports/colab_t4_expanded_continuation_20261030.json)。下一段使用[03a金融衔接入口](DOMAIN_HANDOFF.md)，先在Drive核验母检查点；其新的T4金融执行尚未发生。

## 运行顺序

1. 打开 [01b：扩充语料](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01b_expand_data.ipynb)，CPU 即可；完成 Drive 授权，从第一格顺序运行。
2. 查看 `expansion_summary.json` 的实际独立篇数、去重／排除数量和 token 数；确认 `corpora`、`prepared`、`processed` 三部分备份均 `verified: true`。
3. 打开 [02c：扩充通用预训练](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02c_expanded_pretrain.ipynb)，选择 T4，完成授权后顺序运行。
4. 02c 先测母检查点在原 seed 验证分片上的限量 loss，验证新阶段第 1 步，再训练到新阶段累计 610 步。最后在同一原验证分片上重新测 loss，完成运行快照校验备份。

两个入口都有完整环境安装和当前内核导入检查。01b 在一次 pip 事务中安装项目与 `[data]` 依赖并保留 Colab 的 PyTorch；02c 安装训练所需依赖。无需重做 01a、02a、02b。旧 `03_domain_pretrain.ipynb` 的路径仍不适用于这些版本，不要直接运行。

## 来源和真实新增量

通用候选取同一固定 Wikipedia revision 的更大前缀，默认最多 200,000,000 字节／50,000 篇。金融候选从 Board 官方历史索引及近年日历的实际链接取得 2011–2025 FOMC 纪要，默认最多 128 篇／12,000,000 字节。版权依据、归属、真实公开日与原 HTML 快照保留，详见 [来源核验](research/expanded_corpus_sources_20261011.md)。

扩充候选包括旧 seed 前缀和 2021–2024 纪要，因此来源字节数不是新增量。准备管线先排除旧 seed 全部训练及验证文档的来源身份、重复／词法近重复内容，以及已冻结的任务原文池；剩余文档才分为新增 train/dev。原 seed 验证集保持原样，扩充语料自己的验证集单独记录。

本轮复用 `tokenizer/seed_v1/tokenizer.json` 的原 hash，不建立新词表。正文逐行处理、token 分片使用 `uint16`，训练时 mmap 读取；不会将全量英文 Wikipedia 转为内存列表。JSONL 输出上限不包含 Parquet 缓冲、HTML／恢复缓存、清洗文本和检查点，不能作为全部空间占用的承诺。

实际新语料规模必须由本次 `expansion_summary.json` 给出。少量宏观纪要不能自动满足 25M 独立金融 tokens，重复回放不能算新增文档。01b 另准备不循环源内容的 80% 金融／20% 通用 `domain_mix` 分片，规模受实际金融源限制。本次 02c 不用它启动金融阶段，更不生成资产看多／看空标签；金融阶段的检查点衔接、配置和 Notebook 需另行交付。

## 本地实际准备结果

默认来源限额已在本地实际下载并完成 CPU 构建，详见 [机器可读报告](../reports/expansion_local_validation_20261011.json)。28,486 篇 Wikipedia 和 120 篇 FOMC 候选中排除了旧 seed 的 825 篇通用、32 篇金融文档；10,686 条冻结任务原文纳入排除检查。原 seed 清单和 8192-token 词表 hash 保持不变。

| 新增分片 | 文档数 | tokens |
|---|---:|---:|
| 通用 train | 26,267 | 41,499,364 |
| 通用 validation | 1,394 | 2,235,968 |
| 金融 train | 84 | 972,438 |
| 金融 validation | 4 | 50,999 |

`domain_mix` 为 1,215,545 tokens，其中金融 972,436、通用 243,109，比例为 80/20。它重新组织上述训练源，不能再计为独立新增语料。金融内容是宏观纪要，尚不覆盖推文立场或人工资产标签。

本地构建耗时 544.91 秒；每 0.1 秒采样的进程 RSS 峰值为 361,443,328 字节，约 344.7 MiB。候选／缓存 247,243,919 字节、清洗产物 248,168,041 字节、token 产物 91,966,277 字节，合计 587,378,237 字节，约 560.2 MiB。这些仅为本地 CPU 测量和三类产物的占用，不含旧 seed、模型检查点及外部缓存；采样 RSS 也不是精确分配峰值或 T4 显存。

01b／02c 交付时的 205 项本地测试通过（274.58 秒），12 本 Notebook 检查通过，`pip check` 无损坏依赖，代码单元格已在隔离 Jupyter 内核及本地替代输入上执行。用户另提供 01b 成功构建及三部分校验备份、02c 的 610 步训练和最终备份输出；代理未独立读回 Drive 文件或下载云端权重。这些证据分别记录，不将本地测试当成云端复跑。

## 权重承接和恢复语义

原 `--resume` 要求数据 hash 相同，不能用来把新语料伪装为旧训练流。02c 使用显式 `pretrain_transition.kind = data_expansion`，冻结并校验母检查点的 SHA256、正式预训练身份、至少 300 步进度及 tokenizer。

首次新阶段只加载母模型权重，重新建立 AdamW、调度器、AMP scaler、随机状态和数据游标，阶段更新／token 计数从零开始。报告同时列出母模型 tokens、新阶段 tokens 及沿此模型路径处理的总量。模型架构和词表保持不变。

本段默认 610 次更新，batch 4 × 梯度累积 8 × 序列 512，对应名义 9,994,240 个位置；padding、AMP 跳步等会影响有效计数，实际 token 数按训练报告。新阶段 warmup 50 步、调度跨度 6104 步，当前停止预算 610 步，100M token 上限仍是配置预算。新增通用分片不足以支持本段无回放时明确停止。

新阶段创建检查点后，同版本重复执行只能按新阶段 provenance 严格恢复；不能再次从母模型权重重启。改变 tokenizer、语料、精度或语义配置会报错，需保留旧版本并创建独立实验。每 10 步检查点同步 Drive，报告独立归档。训练、原验证集测量或备份失败时，Notebook 成功标记为 false，后续完整备份单元格停止。

## 保存位置和验收

| 内容 | Colab 工作目录 | Drive 项目目录下 |
|---|---|---|
| 候选和清洗文本 | `data/raw/expansion_v1/{corpora,prepared}` | `expansions/expansion_v1/{corpora,prepared}` |
| token 分片与构建报告 | `data/processed/expansion_v1` | `expansions/expansion_v1/processed` |
| 冻结母检查点 | `checkpoints/expansion_v1_parent` | `expansions/expansion_v1/parent` |
| 新阶段运行快照 | `runs/expansion_v1_pretrain` | `expansions/expansion_v1/run` |
| 定期检查点 | 同上 | `live_checkpoints/expansion_v1_pretrain` |
| 原验证集前后测量／段报告 | 由入口保存 | `expansions/expansion_v1/{retention,report_archive}` |

Drive 项目根目录为 `MyDrive/130U-FinLLM`。原 `seeds/seed_v1` 是只读输入，不被这些入口覆盖。

01b 完成以真实构建报告和三部分校验备份为准；02c 完成以新阶段目标步数、原验证集 before/after、正式段报告及最终 `verified: true` 为准。连接弹窗不等于后台训练已停止：先等连接恢复，确认运行已结束或停止，再从本页第一格恢复，避免并行启动两个训练。

这两个入口的交付不表示全部通用预训练、金融领域训练或后训练完成。后续还需根据实际语料规模和新阶段测量决定下一段，准备匹配版本的金融混合配置，并完成目标资产映射和人工立场标签。

02c的610步记录与其保留集测量独立保存。随后02d严格恢复同一扩充实验到1220步；已读回的新通用验证loss为3.6929，原seed保留子集在该续训前后为4.4032→3.9842，最终7文件／616,109,865字节备份输出为`verified: true`。两种验证集仍分开报告。下一阶段03a核验并冻结1220步模型，首次迁移权重进入新的金融优化状态；不会覆盖原02c测量或原seed成果。
