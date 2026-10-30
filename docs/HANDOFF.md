# 开发与运行交接

项目：130U-FinLLM｜130U 金融大语言模型。仓库：https://github.com/130U/130U-FinLLM 。

## 已冻结

英文、指定资产三分类、自训8K tokenizer、24,255,744参数、随机权重、全参数预训练/金融继续预训练/SFT/DPO、独立校准和测试。参数调整以真实T4测量为依据，不因显存空余直接扩大模型。

## 已交付

原生PyTorch模型、训练/恢复/评测代码、数据格式与审核流程、分片与精确token混合、目标感知稀疏基线、分阶段 Colab Notebook、正式配置、必要测试与CPU工程小实验。实际验证见STATUS。

## 下一步

2026-10-30最新接续：用户已提供完整03a输出，`domain_v1`完成74步／1,211,960有效tokens，金融验证loss 3.2375，seed通用保留集3.9842→4.0574，最终备份输出为7文件／615,896,468字节／`verified: true`。模型路径累计26,115,096有效tokens；本段为有界实验，并未完成原25M金融预算。见[领域记录](../reports/colab_t4_domain_20261030.json)。证据来自用户贴出日志，未独立下载Drive权重。下一步是[接收已完成的人工审核进度](POST_DOMAIN_NEXT.md)，无需重跑03a。

金融衔接完整274项测试通过（557.43秒），14本Notebook检查和依赖检查通过；03a在两个真实隔离内核验证过微型CPU训练、备份和新目录恢复。见[工程记录](../reports/domain_handoff_checks_20261030.json)。后续2000条AI初稿与工作台新增20项检查通过，严格进度导入器新增37项检查通过。用户已在工作台审核但尚未导出，实际确认数量未知；不能将最初空白人工字段计数当作当前用户进度。

2026-10-11 验收：用户先完成 02a 的 61 步、999,424-token T4 试跑，再提供 02b 的连续 63–300 步日志、最终 summary 与备份输出。累计 300 步、4,914,941 有效 tokens，最后训练 loss 5.4151、限量验证 loss 5.5419；第 62→300 步耗时 1150.17 秒、约 3390.06 tokens/秒、PyTorch 分配峰值显存约 1.063 GiB。最终运行快照为 7 个文件、615,936,768 字节、`verified: true`，原始试跑报告保留。详见 `docs/T4_PILOT.md` 与 `reports/colab_t4_continuation_20261011.json`。这是用户提供的运行证据，未独立下载报告/检查点或读回 Drive 文件。首批来源、排除池、8192 词表和四份分片见 `docs/SEED_DATA.md`。

02b 的本段训练和备份已完成，无需重跑。输入位置为 4,915,200，扣除 259 个 padding 后有效训练 tokens 为 4,914,941。通用池约 4.98M tokens，已接近一遍；下一阶段先扩充语料并明确新实验版本。seed 路径独立于 `general/english/pretrain`，03 的默认配置不能直接衔接。200 条 AI 初判表已完成，人工字段仍未确认。

本地扩充准备已验证：新增通用 train 26,267 篇／41,499,364 tokens，金融 train 84 篇／972,438 tokens；新增 validation 为通用 1,394 篇／2,235,968 tokens、金融 4 篇／50,999 tokens。原 seed 的 825＋32 篇已排除，10,686 条任务原文保持冻结，原 8K 词表未重训。CPU 构建耗时 544.91 秒、采样 RSS 峰值约 344.7 MiB，三类产物合计约 560.2 MiB（不含旧 seed、检查点及外部缓存）。01b／02c 初次交付时205项测试、12本Notebook检查通过；加入02d后完整232项测试通过（343.60秒），13本Notebook检查和 `pip check` 通过。见 [本地扩充报告](../reports/expansion_local_validation_20261011.json)、[02d工程检查](../reports/expanded_continuation_checks_20261011.json) 和 [运行说明](DATA_EXPANSION.md)；这些不是新阶段 Colab 运行成绩，云端仍需匹配用户自己的冻结文件。

1. 保留 `seed_v1` 的原始数据、词表、300 步检查点、报告归档和已验证快照，不重做 01a/02a/02b。
2. 保留并复用已完成 01b 的独立 `expansion_v1` 输入和三部分校验备份，不重新下载或构建。该版本已按固定来源排除原 seed／任务原文、去重、文档 train/dev 切分并用原 tokenizer 编码。来源为限量 Wikipedia 和 2011–2025 FOMC；实际新增规模见报告。不能把重复文本计为新增 100M 数据。
3. 用户已完成01b、02c和02d，保留seed300步、原610步归档和1220步最终快照。02d执行证据固定在用户历史提交`e14b85e5660c4bce175341ee4d9da2e3984a12b5`；主分支可运行Notebook保持无输出。CI曾因保存执行输出导致生成器字节检查失败，没有训练数值失败证据。
4. 保留已完成的[03a领域成果](DOMAIN_HANDOFF.md)：Drive`domains/domain_v1/run`、`retention`、`report_archive`和`parent`。当前最终latest SHA为`2a6585bf18ed4753330221292d89b45ab17a52388ab7a0350eaf5d0dc4255c1f`（来自用户after诊断输出）。准备SFT时需重新读回完整快照、匹配报告与诊断，再冻结选定母权重；不重新初始化原阶段或绕过检查。
5. 用户要求AI先填、自己审核后，已扩为[2000条AI初稿与审核工作台](REVIEW_WORKSPACE.md)。初稿有1916条唯一历史来源代码候选、2条多义、82条未匹配；这些不等于当前证券核验。用户已在工作台审核，需导出`review_progress_*.json`才能验收实际进度。使用`scripts/import_review_progress.py`核对原数据SHA和行号；草稿、待查、排除及未核实资产分别保留，产物仍禁用正式训练。SFT/DPO等待来源、资产、事件与分区验收，用户没有选择以弱标签替代人工真值。

## 恢复

- `latest.pt`及其`.json`成对保留。参考DPO的SFT检查点必须保持不变。
- 正式Notebook挂载Drive后，配置会将各次保存的检查点同步至`MyDrive/130U-FinLLM/live_checkpoints/<stage>`。最终数据/模型目录另做带清单的snapshot。
- 新VM先恢复data/tokenizer/runs；必要时恢复较新的live_checkpoints；验证源码、数据、tokenizer、运行时版本后显式`--resume`。
- 不能把CPU smoke权重移入正式训练。精确续训绑定设备类型、AMP、PyTorch/CUDA版本、调度语义和数据哈希；变更后应记录为新实验。
- 不把运行日志、完整语料或大权重提交Git。用户凭据只通过服务正常登录流程使用。
