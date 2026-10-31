# 24M Trial｜首轮语料来源核验

范围：24M Trial 历史来源研究，保留 2026-10-09 的证据、筛选与阶段安排；后续执行结果见[Trial 实验报告](../experiment-results.md)，项目整体见[总览](../../README.md)。

核验日期：2026-10-09。用途是英文从零预训练和金融领域继续预训练；不把金融说明文字自动转换成资产看多／看空标签。来源研究后已执行限量下载，实际记录见文末；尚未据此宣称训练已完成。

## 研究边界与来源

知识深度：working。目标：给出具有版本、来源及再使用依据、可在 Colab 限量导入的第一批真实语料。

- Scope / Object / Subject：从零语言模型语料 / 首轮 1–5M token 工程训练与以后 100M token 预算 / Wikipedia、FineWeb-Edu、Federal Reserve Board 原创英文纪要。
- 类型：descriptive；模型及数据管线已存在，本次核对可执行来源合同。
- Q1：文本的再使用依据是什么？Q2：固定版本、字段和获取入口是什么？Q3：如何限制内存与磁盘？
- Canon：先数据发布者卡片与 API；再 Wikimedia、Federal Reserve Board、Common Crawl 官方条款；最后对固定数据分片和 HTML 直接检查。未采用二手推荐文章。

| 优先级 | 原始来源组 | 核验用途 |
|---|---|---|
| 1 | [Wikimedia HF 数据卡](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)、[数据集 API](https://huggingface.co/api/datasets/wikimedia/wikipedia) | 固定 SHA、英文配置、字段、数据集许可 |
| 1 | [Wikimedia 官方条款 §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content) | 归属、许可、修改及转载要求 |
| 1 | [FineWeb-Edu 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)、[Common Crawl 条款](https://commoncrawl.org/terms-of-use) | 数据库许可与原网页权利的区别 |
| 1 | [Federal Reserve Board 使用说明](https://www.federalreserve.gov/disclaimer.htm) | Board 网站原创资料的公共领域声明及例外 |
| 1 | [FOMC 官方日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm)、[实际纪要样例](https://www.federalreserve.gov/monetarypolicy/fomcminutes20241218.htm) | 真实入口、发布日、正文选择器 |
| 2 | [Hugging Face Datasets streaming 文档](https://huggingface.co/docs/datasets/stream) | 流式迭代、缓冲和分片行为 |

## 证据记录

| ID | 核验发现 | 来源／证据 | 回答 |
|---|---|---|---|
| F1 | `wikimedia/wikipedia` 的当前核验 SHA 为 `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`；英文配置 `20231101.en`，split `train`；四个字符串字段 `id, url, title, text`。数据卡声明 `cc-by-sa-3.0` 和 `gfdl`。 | 固定数据卡及 API 直接读取 | Q1/Q2 |
| F2 | 英文配置包含 6,407,814 条，下载大小 11,630,929,031 字节；首分片 `20231101.en/train-00000-of-00041.parquet` 为 420,296,449 字节，其 LFS SHA256 为 `382e7f6f09e488b24793a7f7cfc659879d5a22da2cf2efec6491665f0c019677`。 | 数据卡和[固定树 API](https://huggingface.co/api/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/20231101.en) | Q2/Q3 |
| F3 | Wikimedia 当前条款使用 CC BY-SA 4.0，与 HF 2023 快照卡片所写 3.0 有版本差异；文本归属可保留原文章 URL，导入来源的额外归属要保留，修改文本应标记修改并保留适用许可。 | 官方条款 §7；这是应保存的版本差异，不自动改写快照许可 | Q1 |
| F4 | FineWeb-Edu SHA 为 `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`；`sample-10BT` 对应 `sample/10BT/*`，有 `text/id/url/date` 等字段。发布者写 ODC-By 1.0，同时指向 Common Crawl 条款；后者明确被抓取内容可另有原站条款、第三方版权。 | [FineWeb-Edu API](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu)、卡片与 Common Crawl §2–3 | Q1/Q2 |
| F5 | Federal Reserve Board 网站说明：除另有标记外，信息为公共领域，应注明 Board 来源；非 Board 图片／图形／其他材料须从第三方取得相应许可，网站链接不扩大这种许可。 | 官方 disclaimer，页面标注更新 2024-08-02，2026-10-09读取 | Q1 |
| F6 | 日历实际给出 2021–2026 纪要链接和发布时间；样例 `fomcminutes20241218.htm` HTTP 200，HTML 126,531 字节，有 `#article` 正文节点；会议日为2024-12-18，纪要实际发布2025-01-08。 | 官方日历与直接 HTML 检查 | Q2 |
| F7 | `streaming=True` 创建 IterableDataset，可迭代获取而无需先转完整本地 Arrow 数据集；Parquet仍会有解码与读取缓冲，不能声称“零内存”。 | Datasets 官方 streaming 文档 | Q3 |

## 筛选结果

Scope filter：排除社媒立场标签、收益回测、付费财经新闻及全文下载所有候选语料；这些不属于本轮无标签语料工作。

Object filter：FineWeb-Edu 不作为默认通过权利审查的语料；ODC 数据库许可不能自动当作每篇原网页的训练许可。只选 `federalreserve.gov` 的 Board 原创正文，不把 `federalreservehistory.org`、各地区银行、第三方引用图片和外链自动视为相同权利范围。Wikipedia 保留原始许可记录及文章归属，不把它称为公共领域。

结论：首轮推荐 **Wikipedia 限量子集作为通用语料 + Federal Reserve Board 原创 FOMC 纪要作为小型金融语料**。这是基于 F1–F7 的实施选择；两者仍存在百科文体、宏观文体与推特资产立场文本的明显分布差异。

## 最小导入方案

### 通用英文

使用现有 `fetch-corpus`，不下载整个英文 Wikipedia。建议首次 `max_documents=5000`，输出 JSONL 硬上限 `20_000_000` 字节，达到任何一个限制即停止。该上限对应多少项目 tokenizer tokens 必须在 tokenize 后测量，不能把字节数直接当 token 数。

```bash
python -m finllm fetch-corpus \
  --dataset wikimedia/wikipedia \
  --revision b04c8d1ceb2f5cd4588862100d08de323dccfbaa \
  --config 20231101.en --split train --text-field text \
  --max-documents 5000 --max-bytes 20000000 \
  --training-rights-reviewed \
  --rights-note "Wikipedia contributors; original article URL retained; dataset card CC-BY-SA-3.0/GFDL; preserve applicable attribution and share-alike notices; see corpus_sources_20261009.md" \
  --output data/raw/wikipedia_en_pilot.jsonl
```

当前 importer 已保留 `source_url`、revision、rights_note、manifest 和文件 SHA256。数据选择按源顺序，因而是可复现的方便子集，不是代表性随机样本；后续扩大训练时固定抽样种子与分片顺序。不要对流式集合调用 `list(dataset)` 或整表 `to_pandas()`。

### 金融英文

建议第一次只从实际日历解析得到的 2021–2024 纪要链接中导入，按会议日期排序；最多 32 篇、输出 JSONL 最多 8MB、每个响应最多 2MB、串行请求间隔至少 1 秒。不要靠日期猜测不存在的网址，不追踪域外链接。按实际公开日记录 `published_at`，不可仅把会议日当公众可见日。这个选择也避免此次取样随着2026年的后续会议增加而悄悄变化。[F6]

抽取 `#article`；保留标题和段落，去除导航、脚注返回链接、名单等不需要的页面结构，并标记这一清理。每条保存 URL、标题、会议日、公开日、抓取UTC时间、原HTML SHA256、正文SHA256、提取器版本、Board出处、公共领域依据URL和版权例外检查结果。对正文中另行标注第三方权利的页面跳过并记日志。不要把声明页复制到训练文本。

纪要属于宏观领域小型起步集，真实 token 数不足 25M 时如实报告，不循环复制来宣称取得 25M 独立语料。之后可单独审核 Board 原创货币政策报告和金融稳定报告作为扩充；其总规模本次未测量。

## 空间与进入训练的条件

以下为工程预算，不是已测量 Colab 峰值：通用输出上限20MB + 金融输出上限8MB + 元数据；流式读取仍需额外临时缓存。`uint16` token 文件净载荷为 `2 × tokens` 字节，1M=2MB、5M=10MB、100M=200MB（十进制），另加索引、manifest、源文本与模型检查点。避免保留全量 HF 下载缓存和每次训练的多余完整检查点。

来源批准只解决输入依据，还须满足项目既有进入条件：

1. 冻结用于立场评估的真实文本排除池，再做去重／泄漏检查；不能因标签尚未准备而用空池冒充完成。
2. 按文档而不是切块划分语言模型 train/dev，tokenizer 仅用 train 文本。
3. 清理后记录实际独立文档数、字节数、token数、分布与重复比例。
4. T4 首次运行先测 24M 模型一个短训练段的峰值显存、吞吐与恢复，再决定扩大预算；不凭本note承诺时长。

## 验证状态和风险

- 已核验：公开 API 的两个不可变 SHA、Wikipedia 字段和首分片大小、官方再使用条款、FOMC 日历／示例正文节点。
- 本次来源研究未运行 tokenizer、T4 或金融判断评估；限量下载的后续执行见下文。
- 仍需处理：真实资产立场标签及其许可、Wikipedia文本当前许可与快照卡片差异的记录、金融语料覆盖不足、实际缓存峰值和训练吞吐。

研究进度：1 Canon完成；2 Charter完成；3 descriptive文件已读、证据表完成；4 Scope filter完成；5 Object filter完成；6 结论与未决项完成。

## 后续真实下载记录

2026-10-09 已执行 `scripts/fetch_seed_corpora.py --output data/raw/seed_v1/corpora`。这次执行使用 `datasets==4.8.5`、`requests==2.34.2`、`beautifulsoup4==4.15.0`；HF 为未登录的公开流式访问，没有下载全量英文库。脚本默认20MB通用上限、10,000篇上限、5MB金融上限、32篇金融上限。

| 本地产物（不提交原文到 Git） | 实际文档数 | 实际字节数 | SHA256 |
|---|---:|---:|---|
| `general_candidate.jsonl` | 825 | 19,985,370 | `cee8dfb34401588ca8ecde6ec8d13d1ffa275efb5e0ca1391499da16e6a1b22f` |
| `finance_candidate.jsonl` | 32 | 1,791,245 | `f864068487758cdcec3142a526c9abcc5da7e92f14f289f201078f34e424f1c6` |

候选 JSONL 合计21,776,615字节。含32篇原HTML、日历HTML、实际链接列表、manifest的整个输出目录26,299,150字节；这是产物空间，未声称包含全进程峰值内存或外部HF缓存。金融链接32条均成功，无跳过；会期2021–2024，另保留真实发布日期，最后一篇2024-12-18纪要发布于2025-01-08。manifest状态`complete`，两个JSONL总哈希经读回一致。

离线测试 `tests/test_fetch_seed_corpora.py` 四项通过：同域实际链接/发布日期、正文抽取/版权标记拒绝、通用JSONL字节上限与文章归属、拒绝覆盖已有输出。未安装可选HTML依赖时其中两项会显式skip，应在需要完整解析覆盖的CI中安装beautifulsoup4。

这些文件仍是 `candidate_only=true`、未标注的原文候选；去重、train/dev划分、真实资产立场排除池、tokenizer和实际token统计由后续步骤完成。
