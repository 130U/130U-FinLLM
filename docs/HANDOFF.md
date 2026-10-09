# 开发与运行交接

项目：130U-FinLLM｜130U 金融大语言模型。仓库：https://github.com/130U/130U-FinLLM 。

## 已冻结

英文、指定资产三分类、自训8K tokenizer、24,255,744参数、随机权重、全参数预训练/金融继续预训练/SFT/DPO、独立校准和测试。参数调整以真实T4测量为依据，不因显存空余直接扩大模型。

## 已交付

原生PyTorch模型、训练/恢复/评测代码、数据格式与审核流程、分片与精确token混合、目标感知稀疏基线、七本Colab Notebook、正式配置、必要测试与CPU工程小实验。实际验证见STATUS。

## 下一步

2026-10-09 进展：用户截图已证明 00 的 T4 小实验、Drive 挂载与备份完成。首批真实来源、排除池、8192 词表与四份分片已在本地生成，见 `docs/SEED_DATA.md` 与 `reports/seed_data_preparation.json`。下一入口为 01a → 02a；seed 路径和 Drive 备份独立于原有 `general/english/pretrain` 路径，不可混用配置。200 条 AI 初判表已完成，人工字段仍未确认。

内置浏览器再次出现 `Monaco: unable to load`，没有代用户运行 01a/02a，不声称 T4 24M 资源试跑通过。现阶段使用整个任务候选池作为预训练排除池，让语料准备可独立于标签审核继续；SFT/DPO 仍等待目标资产和立场审核。

1. 用户已完成第一本 Colab。当前在 Chrome 打开 01a 准备并备份 seed 数据，再打开 02a 运行 24M 短段；这两本会挂载 Drive。
2. 收取summary或报错，核对真实GPU型号、FP16兼容性、版本、内存和恢复。没有访问浏览器/内核时不声称已经代用户运行。
3. 准备和审核真实来源；先冻结评测原文池，再训练正式tokenizer、编码语料。没有数据也可继续完善独立工程测试，但不能宣称完成正式训练。
4. 先运行pretrain_pilot（约1M），保持同一优化器调度和数据版本恢复到100M。SFT/DPO按开发集而非test选择。FinBERT比较、消融/重复seed/组bootstrap等按数据和算力开展。

## 恢复

- `latest.pt`及其`.json`成对保留。参考DPO的SFT检查点必须保持不变。
- 正式Notebook挂载Drive后，配置会将各次保存的检查点同步至`MyDrive/130U-FinLLM/live_checkpoints/<stage>`。最终数据/模型目录另做带清单的snapshot。
- 新VM先恢复data/tokenizer/runs；必要时恢复较新的live_checkpoints；验证源码、数据、tokenizer、运行时版本后显式`--resume`。
- 不能把CPU smoke权重移入正式训练。精确续训绑定设备类型、AMP、PyTorch/CUDA版本、调度语义和数据哈希；变更后应记录为新实验。
- 不把运行日志、完整语料或大权重提交Git。用户凭据只通过服务正常登录流程使用。
