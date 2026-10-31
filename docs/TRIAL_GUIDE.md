# 24M Trial｜公开运行指南

Trial 是 130U-FinLLM 的公开工程版本。代码、默认训练配置和 Notebook 均对应此路径；2.4B 自用版本的架构和方法见 [模型说明](MODEL_CARD.md) 与 [项目介绍](PROJECT_OVERVIEW.md)。

## 快速检查完整链路

要求 Python 3.10 或更高版本。

```bash
python -m pip install -e .
python -m finllm smoke --device cpu --output runs/first-smoke
```

该命令使用项目生成的小型合成样本，训练小配置并检查 tokenizer、预训练、领域训练、SFT、偏好训练接口、恢复、校准和预测。输出保存在指定目录；再次运行请换一个新目录。此入口用于工程检查，24M 配置单独位于 [`model_24m.json`](../configs/model_24m.json)。

## 24M 训练路径

| 阶段 | 代码与文档 |
|---|---|
| 数据字段、资格与分区 | [DATA_GUIDE](DATA_GUIDE.md) · [标注模板](../data/examples/annotation_template.jsonl) |
| 首批原文、词表与分片 | [SEED_DATA](SEED_DATA.md) · [数据准备 Notebook](../notebooks/01a_seed_data.ipynb) |
| 基础训练与扩充 | [模型 pilot](../notebooks/02a_seed_pilot.ipynb) · [扩充训练](../notebooks/02c_expanded_pretrain.ipynb) |
| 金融领域适配 | [领域衔接](DOMAIN_HANDOFF.md) · [领域 Notebook](../notebooks/03a_domain_pretrain.ipynb) |
| 目标立场 SFT | [实体任务](ENTITY_SFT_PILOT.md) · [平衡曝光脚本](../scripts/kaggle_balanced_sft.py) |
| 校准与预测 | [推理导出脚本](../scripts/kaggle_inference_export.py) · [CLI](../src/finllm/cli.py) |

正式任务需要自行准备符合字段与来源要求的数据。现有 headline 实验使用私有材料及领域检查点；公开仓库提供代码和记录，数据与权重由各自工作流管理。

## 命令与配置

```bash
python -m finllm --help
python -m finllm train --help
python -m finllm predict --help
```

[`pretrain.json`](../configs/pretrain.json)、[`domain.json`](../configs/domain.json) 与 [`sft.json`](../configs/sft.json) 展示各阶段配置。运行前按数据指南准备 tokenizer、训练／验证材料及输出路径。阶段承接使用 `init_checkpoint`；同阶段继续训练使用 `resume`，同时恢复优化与数据状态。

## 结果与工程记录

[Trial 实验结果](experiment-results.md) 保留样本分母、逐类指标与运行条件；[工程复盘](ENGINEERING.md) 解释类别学习、语义边界与扩容取舍；[Trial 状态](STATUS.md) 汇总公开路径。`reports/` 的机器可读记录与原实验对应。

## 开发检查

```bash
python -m pip install -e ".[test]"
python -m pytest
python scripts/validate_notebooks.py
```

模型输出包括标签、三类概率、方向分数和复核信息。文本的目标立场与后续资产回报属于不同对象，调用方应按任务定义使用输出。
