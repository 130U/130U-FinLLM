# 130U-FinLLM · 130U 金融大语言模型

An English financial stance language model trained from random weights: pretraining → domain pretraining → full-parameter SFT → DPO → calibration and evaluation.

输入英文短文本和指定资产，输出对该资产的 **看空 / 中性 / 看多**、三类概率、方向分数和复核标记。首版正式配置为 **24,255,744 参数**，自建 8192-token byte-level BPE。代码结构参考 [MiniMind](https://github.com/jingyaogong/minimind/tree/f659b55761b754d306bd140573493a6543cafd7f)，主模型不下载其他模型的预训练权重。

**当前交付是可运行的训练工程，不是已训练完成的金融模型。**用户已在 T4 完成微型流程验证及 Drive 备份。首批真实英文语料、8192-token 自训词表及训练分片已在本地生成；24M 模型的 GPU 预训练与真实金融效果仍待执行。

## 已完成试跑：从这里继续

[01a：下载限量真实语料、冻结评测排除池、训练词表并备份](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb) → [02a：24M 模型约 1M-token GPU 预训练试跑](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb)。两本都默认挂载 Drive，并按 seed 版本独立备份。完整操作与实际产物见 [首批数据执行记录](docs/SEED_DATA.md)。

01a 支持完整性校验后的重复运行及 Drive 恢复，可使用 CPU。旧版出现目录已存在或 tokenizers 依赖冲突时，请按 [Colab 修复说明](docs/COLAB_DEBUG.md) 重新打开最新版，无需删除已有成果。

首批训练数据为 4,976,894 个通用 tokens 和 341,309 个金融 tokens；验证文本另存。全部 10,686 个不同候选任务标题在词表训练前排除。没有将来源情绪标签自动改为资产立场标签。

## 第一次：只运行这个 Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/00_setup_and_smoke.ipynb)

1. 打开上面的 Notebook，登录 Google。需要保留编辑时保存副本到自己的 Drive。
2. 选择 **运行时 → 更改运行时类型 → T4 GPU**。
3. 执行环境单元格。它会固定本次代码提交，保留 Colab 自带的 PyTorch/CUDA，安装其余小型依赖。
4. 需要备份时，把 `MOUNT_DRIVE` 改为 `True`，在 Google 界面完成 Drive 授权。无需向他人发送密码或访问 token。
5. 运行 smoke 单元格，再运行备份单元格。它只使用项目自编样本，不下载大型语料、不启动 100M-token 正式训练。
6. 将 `summary.json` 或报错日志带回工作对话，继续检查真实 T4 环境。

Colab Notebook 分享不共享正在运行的虚拟机。`/content` 为临时工作盘，Drive 存持久化成果。新运行时需重新安装环境并恢复文件。[Google 官方说明](https://research.google.com/colaboratory/faq.html)

## 后续阶段

| Notebook | 用途 | 前提 |
|---|---|---|
| [00_setup_and_smoke](notebooks/00_setup_and_smoke.ipynb) | 环境、小模型全流程、恢复、校准与预测 | 仅需运行环境 |
| [01_prepare_data](notebooks/01_prepare_data.ipynb) | 审核数据、分区、排除评测内容、自建词表、分片、80/20 token 混合 | 获准使用的正式原文与目标标签 |
| [01a_seed_data](notebooks/01a_seed_data.ipynb) | 限量真实来源、完整候选池排除、自训 8K 词表、版本备份 | 可联网；标签审核可以并行 |
| [02a_seed_pilot](notebooks/02a_seed_pilot.ipynb) | 首批数据的 24M、61 次更新预训练试跑 | 01a 完成、T4 和 Drive |
| [02_pretrain](notebooks/02_pretrain.ipynb) | 从随机权重开始通用预训练 | 正式 tokenizer、训练/验证分片 |
| [03_domain_pretrain](notebooks/03_domain_pretrain.ipynb) | 金融继续预训练 | 通用预训练检查点及金融混合分片 |
| [04_sft](notebooks/04_sft.ipynb) | 全参数目标资产立场监督学习 | 训练标签和开发集 |
| [05_dpo](notebooks/05_dpo.ipynb) | 冻结 SFT 参考模型的偏好训练 | 审核过的偏好对 |
| [06_evaluate_and_predict](notebooks/06_evaluate_and_predict.ipynb) | 开发集选型、独立校准、最终测试及批量预测 | 完成训练、分区和选型 |

请先读 [数据准备说明](docs/DATA_GUIDE.md) 与 [执行计划](docs/PLAN.md)。正式数据尚未随仓库分发；空缺材料会明确报错，不会用假数据冒充正式训练。

## 本地快速验证

Python 3.10+，PyTorch 2.5+。有可用的 CUDA PyTorch 时保留该安装；CPU 也可运行小实验。

```bash
python -m venv .venv
# 激活虚拟环境后：
python -m pip install -e '.[test]'
python -m pytest -q
python -m finllm environment
python -m finllm smoke --output runs/smoke --device cpu
```

重复执行时选用新的输出目录，避免覆盖既有证据。该小实验包含四阶段各两次更新和一次 DPO 完整恢复；合成模板存在重复，不能用其中的评测数字推断真实金融能力。

## 正式模型与训练

| 项目 | 初始规格 |
|---|---|
| 架构 | Dense decoder-only Transformer，GQA、RoPE、Q/K RMSNorm、共享 embedding |
| 层数 / 隐藏维度 / FFN | 6 / 512 / 1664 |
| Query / KV heads | 8 / 4 |
| 总词表 / 上下文 | 8192（含全部特殊 token）/ 512 |
| 参数 | 24,255,744 |
| T4 起点 | FP16 AMP + GradScaler，FP32 主权重，micro-batch 4，梯度累积 8 |
| 初始 token 预算 | 100M 通用；25M 金融阶段＝20M 金融＋5M 通用回放 |

先执行 `configs/pretrain_pilot.json` 的 1M-token 试跑，再以同一调度和检查点继续 `configs/pretrain.json`。这是正式 100M 预算的前段，不另算一个从其他权重开始的模型。真实速度和峰值以 `run_summary.json` 为准。

```bash
python -m finllm train --config configs/pretrain_pilot.json
python -m finllm train --config configs/pretrain.json --resume runs/pretrain/latest.pt
python -m finllm train --config configs/domain.json
python -m finllm train --config configs/sft.json
python -m finllm train --config configs/dpo.json
```

若按 token 上限截断了最后一批，精确恢复会拒绝直接扩展该 token 上限；pilot 配置采用更新步数停止，避免这种不完整批次状态。正式短跑后检查有效 token 数再继续。OOM 时先减少批次；配置、环境或数据变化可能需要新实验，而非伪称精确续训。

## 输出含义

模型在固定答案位置输出三个专用标签 logits，经最终模型的温度校准后计算：

```text
sentiment_score = 100 * (P(bullish) - P(bearish))
confidence = max(P(bearish), P(neutral), P(bullish))
```

分数范围为 [-100,100]，表示文本对目标资产的方向倾向；不是收益率、股价上涨概率或已验证的交易信号。同一条 `Long $AAPL, short $TSLA.` 应按两个 target 分别输入。无法核验目标关联、语言不支持或输入过长时输出空分数及复核原因；低置信度不等于中性。

```bash
python -m finllm predict --checkpoint runs/sft/best.pt --tokenizer tokenizer/english \
  --input data/raw/predict.jsonl --output reports/private/predictions.jsonl
```

预测输入最少为 `{"text":"I am bullish on $AAPL.","target":"AAPL"}`。只有审核过的明确公司别名才能额外提供 `target_aliases`。

## 复现与数据边界

- 预训练输入只移位一次。SFT/DPO 只监督答案标签，不用 prompt、padding 或 EOS 凑损失。
- train / development / calibration / test 分离；calibration 再分温度拟合和复核阈值选择。Tokenizer 和预训练都排除封存评测原文。
- 保存权重、Adam、scheduler、GradScaler、随机状态、数据游标、配置和哈希。`latest.pt` 与其 `.json` 完整性标记必须一起保留。
- 仅加载自己生成或明确可信的 PyTorch 检查点；其完整优化器状态需要反序列化 Python 对象。
- `finllm sync` 使用文件哈希和完成标记备份/恢复。大数据、检查点、运行日志和个人文件不进入 Git。
- DPO 是需要做的学习实验，最终采用 SFT 还是 DPO 由开发集决定，不能默认 DPO 更好。

验证状态和已知限制见 [STATUS](docs/STATUS.md)。代码采用 Apache-2.0，参考来源及改造范围见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。数据和以后训练出的权重，其分发条件需要另外核对。
