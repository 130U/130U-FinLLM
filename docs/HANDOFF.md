# 开发与运行交接

项目：130U-FinLLM｜130U 金融大语言模型。仓库：https://github.com/130U/130U-FinLLM 。

## 已冻结

英文、指定资产三分类、自训8K tokenizer、24,255,744参数、随机权重、全参数预训练/金融继续预训练/SFT/DPO、独立校准和测试。参数调整以真实T4测量为依据，不因显存空余直接扩大模型。

## 已交付

原生PyTorch模型、训练/恢复/评测代码、数据格式与审核流程、分片与精确token混合、目标感知稀疏基线、分阶段 Colab Notebook、正式配置、必要测试与CPU工程小实验。实际验证见STATUS。

## 下一步

2026-10-11 验收：用户先完成 02a 的 61 步、999,424-token T4 试跑，再提供 02b 的连续 63–300 步日志、最终 summary 与备份输出。累计 300 步、4,914,941 有效 tokens，最后训练 loss 5.4151、限量验证 loss 5.5419；第 62→300 步耗时 1150.17 秒、约 3390.06 tokens/秒、PyTorch 分配峰值显存约 1.063 GiB。最终运行快照为 7 个文件、615,936,768 字节、`verified: true`，原始试跑报告保留。详见 `docs/T4_PILOT.md` 与 `reports/colab_t4_continuation_20261011.json`。这是用户提供的运行证据，未独立下载报告/检查点或读回 Drive 文件。首批来源、排除池、8192 词表和四份分片见 `docs/SEED_DATA.md`。

02b 的本段训练和备份已完成，无需重跑。输入位置为 4,915,200，扣除 259 个 padding 后有效训练 tokens 为 4,914,941。通用池约 4.98M tokens，已接近一遍；下一阶段先扩充语料并明确新实验版本。seed 路径独立于 `general/english/pretrain`，03 的默认配置不能直接衔接。200 条 AI 初判表已完成，人工字段仍未确认。

本地扩充准备已验证：新增通用 train 26,267 篇／41,499,364 tokens，金融 train 84 篇／972,438 tokens；新增 validation 为通用 1,394 篇／2,235,968 tokens、金融 4 篇／50,999 tokens。原 seed 的 825＋32 篇已排除，10,686 条任务原文保持冻结，原 8K 词表未重训。CPU 构建耗时 544.91 秒、采样 RSS 峰值约 344.7 MiB，三类产物合计约 560.2 MiB（不含旧 seed、检查点及外部缓存）。01b／02c 初次交付时205项测试、12本Notebook检查通过；加入02d后完整232项测试通过（343.60秒），13本Notebook检查和 `pip check` 通过。见 [本地扩充报告](../reports/expansion_local_validation_20261011.json)、[02d工程检查](../reports/expanded_continuation_checks_20261011.json) 和 [运行说明](DATA_EXPANSION.md)；这些不是新阶段 Colab 运行成绩，云端仍需匹配用户自己的冻结文件。

1. 保留 `seed_v1` 的原始数据、词表、300 步检查点、报告归档和已验证快照，不重做 01a/02a/02b。
2. 保留并复用已完成 01b 的独立 `expansion_v1` 输入和三部分校验备份，不重新下载或构建。该版本已按固定来源排除原 seed／任务原文、去重、文档 train/dev 切分并用原 tokenizer 编码。来源为限量 Wikipedia 和 2011–2025 FOMC；实际新增规模见报告。不能把重复文本计为新增 100M 数据。
3. 用户已完成 01b 及 T4 的 02c。扩充阶段 610 步／9,994,240 tokens，新通用验证 loss 4.1311，原 seed 保留子集 after loss 4.4032；最终 7 文件／615,996,721 字节快照显示 `verified: true`。本轮 before 数值未提供，不计算变化幅度；代理未独立读回 Drive。记录见 `reports/colab_t4_expansion_20261011.json`。下一入口是 02d：先完整归档 610 步成果，再严格恢复优化器／调度器／AMP／RNG／数据游标，验证 611 步并继续至扩充累计 1220 步。原 02c 测量不覆盖；02d 新 Colab 段仍待执行。见 `docs/EXPANDED_CONTINUATION.md`。
4. 本地已验证 01b 生成独立 `expansion_v1/domain_mix`，共 1,215,545 tokens，按 80% 金融／20% 通用取得源内容且不循环复制；这是新增训练源的重排，不能再次计为独立新增数据。金融阶段还需承接新阶段选定检查点的配置与 Notebook，重新建立该阶段优化器／调度，不能原样运行旧 03。
5. SFT/DPO 等待资产映射和人工标签，按开发集而非 test 选择；后续开展 FinBERT 比较、消融与重复 seed。此时不自动启动金融长训练。

## 恢复

- `latest.pt`及其`.json`成对保留。参考DPO的SFT检查点必须保持不变。
- 正式Notebook挂载Drive后，配置会将各次保存的检查点同步至`MyDrive/130U-FinLLM/live_checkpoints/<stage>`。最终数据/模型目录另做带清单的snapshot。
- 新VM先恢复data/tokenizer/runs；必要时恢复较新的live_checkpoints；验证源码、数据、tokenizer、运行时版本后显式`--resume`。
- 不能把CPU smoke权重移入正式训练。精确续训绑定设备类型、AMP、PyTorch/CUDA版本、调度语义和数据哈希；变更后应记录为新实验。
- 不把运行日志、完整语料或大权重提交Git。用户凭据只通过服务正常登录流程使用。
