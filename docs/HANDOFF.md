# 开发与运行交接

项目：130U-FinLLM｜130U 金融大语言模型。仓库：https://github.com/130U/130U-FinLLM 。

## 已冻结

英文、指定资产三分类、自训8K tokenizer、24,255,744参数、随机权重、全参数预训练/金融继续预训练/SFT/DPO、独立校准和测试。参数调整以真实T4测量为依据，不因显存空余直接扩大模型。

## 已交付

原生PyTorch模型、训练/恢复/评测代码、数据格式与审核流程、分片与精确token混合、目标感知稀疏基线、分阶段 Colab Notebook、正式配置、必要测试与CPU工程小实验。实际验证见STATUS。

## 下一步

2026-10-11 验收：用户先完成 02a 的 61 步、999,424-token T4 试跑，再提供 02b 的连续 63–300 步日志、最终 summary 与备份输出。累计 300 步、4,914,941 有效 tokens，最后训练 loss 5.4151、限量验证 loss 5.5419；第 62→300 步耗时 1150.17 秒、约 3390.06 tokens/秒、PyTorch 分配峰值显存约 1.063 GiB。最终运行快照为 7 个文件、615,936,768 字节、`verified: true`，原始试跑报告保留。详见 `docs/T4_PILOT.md` 与 `reports/colab_t4_continuation_20261011.json`。这是用户提供的运行证据，未独立下载报告/检查点或读回 Drive 文件。首批来源、排除池、8192 词表和四份分片见 `docs/SEED_DATA.md`。

02b 的本段训练和备份已完成，无需重跑。输入位置为 4,915,200，扣除 259 个 padding 后有效训练 tokens 为 4,914,941。通用池约 4.98M tokens，已接近一遍；下一阶段先扩充语料并明确新实验版本。seed 路径独立于 `general/english/pretrain`，03 的默认配置不能直接衔接。200 条 AI 初判表已完成，人工字段仍未确认。

1. 保留 `seed_v1` 的原始数据、词表、300 步检查点、报告归档和已验证快照，不重做 01a/02a/02b。
2. 下一交付先准备扩充语料 Notebook：来源清单、任务原文排除、去重、训练与诊断验证划分、现有 tokenizer 编码和新版本备份。不能将同一批 4.98M 数据重复约 20 遍写成新增 100M 数据。
3. 扩充通用训练时新增明确的跨数据版本权重承接路径与验证：现有 `--resume` 不允许数据哈希变化，`stage=pretrain` 也不允许 `init_checkpoint`；不能关闭 provenance 检查或静默改为随机初始化。新实验应记录母检查点及优化器／调度策略。
4. 金融阶段另准备匹配 seed 的 80/20 金融／通用混合分片、配置与 Notebook。当前实现允许以验证选择的通用 `best.pt` 初始化 domain 阶段，重新建立该阶段优化器／调度，不能原样运行旧 03。
5. SFT/DPO 等待资产映射和人工标签，按开发集而非 test 选择；后续开展 FinBERT 比较、消融与重复 seed。此时不自动启动金融长训练。

## 恢复

- `latest.pt`及其`.json`成对保留。参考DPO的SFT检查点必须保持不变。
- 正式Notebook挂载Drive后，配置会将各次保存的检查点同步至`MyDrive/130U-FinLLM/live_checkpoints/<stage>`。最终数据/模型目录另做带清单的snapshot。
- 新VM先恢复data/tokenizer/runs；必要时恢复较新的live_checkpoints；验证源码、数据、tokenizer、运行时版本后显式`--resume`。
- 不能把CPU smoke权重移入正式训练。精确续训绑定设备类型、AMP、PyTorch/CUDA版本、调度语义和数据哈希；变更后应记录为新实验。
- 不把运行日志、完整语料或大权重提交Git。用户凭据只通过服务正常登录流程使用。
