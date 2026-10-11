# 首批真实数据执行记录

2026-10-09。以下下载、划分、词表训练和编码均已在本地执行；没有把待执行的 Colab 阶段写成已完成。

## 实际产物

| 项目 | 已执行结果 |
|---|---:|
| Wikipedia 固定版本子集 | 825 篇，19,985,370 字节 JSONL |
| 美联储原创 FOMC 纪要快照 | 32 篇，1,791,245 字节 JSONL |
| 通用训练／验证 | 794／31 篇；4,976,894／135,705 tokens |
| 金融训练／验证 | 31／1 篇；341,309／9,209 tokens |
| 候选任务原文排除池 | 10,686 个不同标题，覆盖 10,753 源记录 |
| 自训 byte-level BPE | 8192 tokens，含 6 个特殊 token |
| 四份 token 分片净载荷 | 10,926,234 字节，约 10.9 MB |
| 实际来源、快照、中间文件、词表与分片 | 约 81 MB，另有下载库缓存和依赖 |

数据保存在本地 Git 忽略目录 `data/raw/seed_v1/`、`data/processed/seed_v1/`、`tokenizer/seed_v1/`。审查表在 `data/raw/sentfin_review_v1/`。GitHub 仅发布脚本、空输出 Notebook 和汇总证据，未发布整库文本或模型权重。

来源与依据：[Wikipedia 固定版本数据卡](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)、[Federal Reserve Board 说明](https://www.federalreserve.gov/disclaimer.htm)、[SEntFiN 作者仓库固定版本](https://github.com/pyRis/SEntFiN/tree/eba43610fe3cb57a3e5773cd6c037da1f8992ad5)。细节、限制及来源筛选见 [通用／金融语料核查](research/corpus_sources_20261009.md)和[任务数据核查](research/stance_sources_20261009.md)。

通用样本是流式前缀，不能声称代表整个 Wikipedia。FOMC 是宏观政策文体，与推文存在分布差异；只有一篇金融验证文档，仅作起步诊断。首批规模足够开展约 1M-token 资源试跑；尚未取得 100M＋25M 独立语料，不把循环回放称为新增数据。

## 数据处理与验证

先冻结所有候选任务原文，再按来源文档哈希分出语言模型验证文档；用 SQLite 全局精确去重及已有 SimHash 排除词法近重复。词表只使用训练文本，最多 12 MB JSONL。逐条编码为 uint16 分片，训练时才读取当前批次并转换 int64。上述步骤不要求整库驻留内存，但 tokenizer 的词频表及排除索引仍会占用 RAM。

已验证实际 24,255,744 参数模型能够对这批真实 tokens 做一次 CPU 前向与反向，loss 和梯度均有限；没有执行优化器更新或保存权重，也不据此推断金融能力。证据见 `reports/seed_model_input_check.json` 与 `reports/seed_data_preparation.json`。

输入脚本拒绝未完成的下载清单、哈希不符和空验证分区。已有完整成果通过清单、配置和文件哈希验证后只读复用；不完整成果保留并报错。Notebook 子进程输出逐行显示。两本 seed Notebook 的备份目录是 `MyDrive/130U-FinLLM/seeds/seed_v1/`，不会替换先前 smoke 或其他 seed 的汇总清单。

## 在 Colab 接着执行

1. 打开 [01a_seed_data.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb)，保存副本，按顺序运行。它先恢复并校验已有成果；首次运行才下载限量来源、建立排除池、训练自己的词表、编码并备份。它可使用 CPU，不会启动模型训练。
2. 完成后打开 [02a_seed_pilot.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb)，选择 T4，按顺序运行。它恢复版本快照，从随机权重开始预训练 61 次更新；已有检查点时先校验并恢复进度。
3. 这段约 1M-token 预训练结束后，读取上方 GPU 型号和 `run_summary.json` 的实际 tokens、吞吐、峰值显存及验证 loss，再决定扩充数据与下一段预算。2026-10-11 用户试跑结果已收到，见 [T4 试跑记录](T4_PILOT.md)。下一入口为 [02b_seed_continue.ipynb](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02b_seed_continue.ipynb)：检查恢复后继续到累计 300 步。保持原调度不意味着现在就自动跑完 100M。

Colab 复现会记录自己的来源快照和 tokenizer 哈希；网页可能更新、库版本也可能不同，不能把本地哈希无条件当作未来 Colab 哈希。续训只能使用与原检查点匹配的那一套输入。重复执行 01a 会校验复用；需要重建时才选择新版本目录，不能覆盖已有 tokenizer。旧版重复执行及依赖冲突的处理见 [Colab 修复说明](COLAB_DEBUG.md)。

01a/02a 由用户在自己的 Colab 中执行。2026-10-11 用户截图显示 24M 模型完成 999,424-token 试跑及 Drive 备份；这一结果不代表完整预训练预算、金融后训练或云端恢复已完成。代理尚未独立读取云端检查点。

## 200 条立场审核

`review_200.csv` 是原始审核表；`review_200_ai_proposals.csv` 是逐条 AI 初判版，每条附中文理由。原始情绪标签、AI 建议、人工确认分列保存，原表未修改。AI 初判为看多 22、看空 16、中性 122、排除 40；它们不是人工真值，也不能用来报告准确率。

审核时填写 `asset_identifier`、`human_stance`、`human_reviewed`：资产代码不清楚时先留空并写说明；`human_stance` 可为 bullish、bearish、neutral、exclude。已有涨跌事实默认中性，明确投资方向或前景才给看多／看空。原始发布数据中监管机构等实体也可能有标签，不能都当可投资目标。

按计划，正式 SFT/DPO 需要确认标签规则及资产映射，之后扩充合格样本、按事件／重复组分区。当前本项目人工确认条数仍为 0。AI 初判只减少审核工作量，不替用户虚构“人工已确认”。
