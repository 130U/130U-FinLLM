# 24M Trial｜英文资产立场数据核查 — 2026-10-09

范围：24M Trial 历史来源研究，保留首次候选池筛选时的结论；后续人工审核与 SFT 进展见[Trial 实验报告](../experiment-results.md)，项目整体见[总览](../../README.md)。

本轮使用 Research 技能要求的一手来源核查：能否取得有明确使用条件、包含目标实体、可以映射为资产立场的数据。结论是：**先采用 SEntFiN 作为人工审核候选池，并在预训练前排除整个池的原文；还不能把来源的 positive/negative 自动当成本项目 bullish/bearish 标签。** 这使预训练准备可以继续，资产映射与标签审核并行进行。

## 选定候选：SEntFiN 1.0

[原作者仓库](https://github.com/pyRis/SEntFiN)由 GitHub 账户 pyRis 维护；[GitHub 用户元数据](https://api.github.com/users/pyRis)返回姓名 Rishu Kumar，与论文作者一致。[固定版本 LICENSE](https://github.com/pyRis/SEntFiN/blob/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE)是 MIT，版权署名为 Ankur Sinha、Satishwar Kedas、Rishu Kumar、Pekka Malo。仓库首页明确发布 dataset and codes，根目录除 README、LICENSE 外是两份数据文件。这里的许可证据来自作者发布仓库，不能据此替新闻原始出版方作额外权利保证；本轮用于本地研究候选审核，保留版权声明，不把原文提交到项目 GitHub。

[作者论文 §3](https://arxiv.org/html/2305.12257)说明：样本来自 2011–2015 年印度财经新闻，三名人工标注者判断实体相关金融情绪，最后采用共识及少量作者裁决。论文报告 10,753 条新闻、14,404 个实体标签。其 neutral 也包括监管机构等不适用金融情绪的实体。因此，“原作者人工标注”与“本项目人工确认资产立场”必须使用不同字段；新闻情绪与明确投资方向并非天然等价。

固定 revision：`eba43610fe3cb57a3e5773cd6c037da1f8992ad5`。

| 文件 | 固定下载地址 | 字节数 | SHA256 |
|---|---|---:|---|
| 主数据 | [SEntFiN.csv](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/SEntFiN.csv) | 1,118,162 | `570af7ba58f724f3fcc414dd5ba80da5b1a3b342134d76c6ebdf53a32c54a92f` |
| 实体名称表 | [entity_list_comprehensive.csv](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/entity_list_comprehensive.csv) | 90,424 | `6ace045edc54c90bc0695ee5a78afec9f7a5294844ccfd0c9e85a14aeea4916b` |
| 许可 | [LICENSE](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/LICENSE) | 1,110 | `961d32986ac3fa2fe2e55ebc7bbae1ba8d632412e98493d73bdf59f0ad8fc65f` |
| 简介 | [README.md](https://raw.githubusercontent.com/pyRis/SEntFiN/eba43610fe3cb57a3e5773cd6c037da1f8992ad5/README.md) | 48 | `543e1779ccb4e13b23107b3c4443891f581041d56adf676daa8db173c730e03a` |

本地独立读取得到：CSV 字段为 `S No., Title, Decisions, Words`；共有 10,753 行、10,686 个完全不同的新闻文本，67 行是重复文本。`Decisions` 是 Python 风格的实体到 `positive/negative/neutral` 字典字符串，13 行因实体名称中的未转义撇号而不符合 Python 字面量语法。这些行从审核抽样池隔离，仍将其新闻原文加入排除池；其余行能解析出 14,371 个目标标签。没有伪造日期，也没有将实体名称猜成股票代码。源 CSV 不提供逐条可靠日期或已验证证券标识。

## 已执行的准备

新增 `scripts/prepare_sentfin_review.py`，只依赖 Python 标准库。下载模式最多接收四个固定文件合计 1,209,744 字节，每个文件用固定大小及 SHA256 校验；已有审核输出目录不覆盖。执行方式：

```bash
python scripts/prepare_sentfin_review.py \
  --source-dir data/raw/candidates/sentfin \
  --output data/raw/sentfin_review_v1 --download
```

本轮本地已下载原文件并运行脚本；文件位于 Git 忽略的 `data/raw/`，不进入公共仓库。

输出为：

- `review_200.csv`：UTF-8 BOM；200 个不同新闻文本，原 negative/neutral/positive 分别 67/67/66，其中 85 条多实体新闻。按原标签分层、seed=130，不能视为总体简单随机样本。
- `task_exclusions.jsonl`：10,686 个完全不同原文，保留所有 10,753 行来源 ID 的归属，包括未抽样及语法异常行。这份文件用于排除 tokenizer/预训练泄漏，与是否已经取得本项目标签无关。
- `manifest.json`：源版本、哈希、数量、隔离行、抽样数量和产物校验值。
- `SOURCE_LICENSE.txt`、`review_instructions.md`：原版权许可及审核规则。

审核表包含 `id,text,source_entity,source_sentiment,source_entity_count,source_entities,entity_is_literal_substring,proposed_stance,asset_identifier,human_stance,human_reviewed,notes`。`proposed_stance`、`asset_identifier`、`human_stance` 全部为空，`human_reviewed` 全部为 false；没有产生任何人工确认或自动猜测的资产立场标签。

人工应先完成目标资产确认，并按本项目规则判断 bullish/bearish/neutral，或记录 exclude 与理由。监管机构、目标不清、讽刺及归属不明案例不能硬塞 neutral。原文只是报告过去涨跌时，不能仅因原来源标签为 positive/negative 就自动产生投资立场。首批 200 条用于识别标签口径及资产映射问题，之后再扩充，不足以宣称金融效果已通过验证。若只有一名审核者，应如实报告单人审核。

专门测试验证了全部候选原文隔离、异常行原文仍保留、未伪造人工标签、抽样输出可重现、旧审核文件不覆盖、源哈希变更时拒绝导入：`tests/test_sentfin_review.py`，本地 **3 passed**。

## 其他来源为何没有直接进入正式训练

| 来源 | 已核查一手证据 | 本轮处理 |
|---|---|---|
| SemEval 2017 Task 5 | [官方数据页](https://alt.qcri.org/semeval2017/task5/index.php?id=data-and-tools)要求下载前填写 data release agreement，gold standard 另需表格申请。[原任务论文](https://aclanthology.org/S17-2089/)确实按公司/股票评估 -1 bearish 到 +1 bullish。 | 定义最接近，但本轮没有已完成来源协议的证据，未下载、未代签、未通过镜像规避。 |
| FiQA 2018 Task 1 | [官方主页](https://sites.google.com/view/fiqa/home)给出 `sentence/info/target/sentiment_score/aspects` 格式，分值 -1 到 1；Train/Test Data 段各写明 non-commercial use。 | 有非商业研究说明和目标字段，但原链接取数失败；不把镜像许可标签当成更强的原始权利证据。 |
| Financial PhraseBank | [Pekka Malo 上传的原始发布页](https://www.researchgate.net/publication/251231364_FinancialPhraseBank-v10)可确认来源作者和压缩包。原下载链接本轮未成功取得；常见镜像将其标为 CC BY-NC-SA 3.0。 | 未仅凭镜像许可直接导入。即使取得，仍需要新增实体与资产映射，原格式是 sentence/sentiment。 |
| zeroshot/twitter-financial-news-sentiment | [发布者数据卡](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)标 MIT，说明来自 Twitter API，数据字段仅 text/label；卡片陈述与页面计数不同。 | 未下载。缺指定目标字段；发布者 MIT 标签不能消除上游 X 使用条件问题。 |

FiQA 官方 Task 1 训练链接为 [Drive 文件 1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch](https://drive.google.com/open?id=1icRTdnu8UcWyDIXtzpsYc2Hm6ACHT-Ch)，本轮直接请求返回 404；测试链接为 [Drive 文件 1t3E2D7sBD3rG-OHNkvu2Fk7tbMzrMws0](https://drive.google.com/open?id=1t3E2D7sBD3rG-OHNkvu2Fk7tbMzrMws0)，仅确认官方链接，不声称已取得内容或确定数量。不能把 FiQA Task 2 的问答语料当作 Task 1 立场数据。

当前 [X Developer Agreement，2026-04-27 版本，III.A(k)](https://docs.x.com/developer-terms/agreement)对 foundation/frontier model 训练或微调设有限制；[Developer Policy 的 Content redistribution](https://docs.x.com/developer-terms/policy)也限制第三方数据再分发。该事实足以支持本项目不默认把 API 来源镜像视作已核查训练许可；这里不对任意旧数据集或法律例外作全面判断。本轮没有抓取或重新获取 X 帖子。

## 2026-10-09 的阶段判定

Q1（来源条件）：SEntFiN 作者仓库有明确 MIT 发布许可及归属，固定小文件已取得。Q2（目标及标签）：来源提供实体相关人工情绪；在该研究时点，本项目的资产映射、投资立场语义还待人工审核。Q3（数量）：当时的候选规模足以开始 200 条口径核验，但实际可用资产立场样本量未知。该时点允许继续预训练语料准备，SFT/DPO 标签尚未验收；后续已完成字面实体任务审核与 SFT，其结果不追溯改写本次来源判定。
