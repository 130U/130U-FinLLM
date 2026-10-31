# 24M Trial｜扩充语料来源核验与获取合同

范围：24M Trial 历史来源研究，本页下载规模及获取合同属于 `expansion_v1`；后续训练结果见[扩充记录](../DATA_EXPANSION.md)，项目整体见[总览](../../README.md)。

核验日期：2026-10-11。承接 `seed_v1` 已完成的 300 步训练；本文件只记录新增候选语料的来源、获取边界和验证，不证明新的 Colab 训练已完成。

## 来源和选择

本轮继续使用首轮已审查的两个来源：固定英文 Wikipedia 快照与 Federal Reserve Board 原创 FOMC 纪要。增加 Wikipedia 的限量前缀，并把 FOMC 会期扩展到 2011–2025。来源顺序固定，不将方便子集描述为随机或代表性抽样。

| 来源 | 一手证据 | 本轮执行合同 |
|---|---|---|
| `wikimedia/wikipedia` | [固定数据卡](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md)、[固定原始数据卡](https://huggingface.co/datasets/wikimedia/wikipedia/resolve/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) | 固定 revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`，配置 `20231101.en`，流式迭代 `train` 的 `id/url/title/text`；默认 JSONL 上限 200,000,000 字节、最多 50,000 篇、单篇正文不超过 1MB。 |
| FOMC 近年纪要 | [官方日历](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) | 从实际 HTML 链接取得 2021–2025 会期，并保存页面中的真实发布日期；不猜纪要网址，不把会期当作公开日。 |
| FOMC 历史纪要 | [官方历史年份索引](https://www.federalreserve.gov/monetarypolicy/fomc_historical_year.htm)、[2011 年页面](https://www.federalreserve.gov/monetarypolicy/fomchistorical2011.htm)、[2020 年页面](https://www.federalreserve.gov/monetarypolicy/fomchistorical2020.htm) | 从索引中实际存在的年份链接取得 2011–2020 页面，再读对应的纪要 HTML 链接；默认共最多 128 篇、金融 JSONL 上限 12,000,000 字节。 |

官方当前日历仅覆盖近年会期，单独重复使用该页不足以得到 2011–2020 纪要。真实 2011 年纪要使用旧网页结构 `#printThis.generalContentText` 和 `#leftText`，新网页使用 `#article`。新提取器分别检验这两个结构；旧结构还必须具有明确的 FOMC 纪要标题。未通过结构检查的文章被记录为跳过，不能退回抓取整个页面。[实际 2011 年纪要](https://www.federalreserve.gov/monetarypolicy/fomcminutes20110126.htm)

## 再使用依据和归属

2026-10-11 直接读取固定 Wikipedia 原始数据卡，HTTP 200，130,835 字节；卡片记录 `cc-by-sa-3.0`、`gfdl`。Wikimedia 当前条款使用 CC BY-SA 4.0，并允许通过文章 URL 保留归属。两者的版本差异继续保存在每条记录的 `license` 与 `rights_note`，不将其改写为公共领域。原文章 URL、快照 revision、正文 hash 和抓取时间均保留。[Wikimedia 官方条款 §7](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content)

Federal Reserve Board 声明其网站信息除另有标识外属于公共领域，并要求注明 Board 来源；非 Board 材料及外部链接不自动获得同样的许可。因此本轮只取 Board 网站上的原创 FOMC 正文，不追踪外链、不取图片、PDF 或其他会议材料；正文含明确版权标记时跳过并保存理由。每条记录保留 Board URL、公开日、原 HTML hash、正文 hash、提取说明及公共领域依据。[Board 网站说明](https://www.federalreserve.gov/disclaimer.htm)

本轮没有抓取 X/Twitter、收费财经文章或新增商业数据集，也没有从纪要生成看多／看空标签。上述来源文体仍偏百科和宏观政策，资产立场监督数据需另行审查。

## 内存、空间和中断恢复

入口是 `scripts/fetch_expansion_corpora.py --output data/raw/expansion_v1/corpora`。输出合同为 `general_candidate.jsonl`、`finance_candidate.jsonl` 和 schema 1 `manifest.json`，兼容既有来源快照校验。只有两部分均校验成功才原子写入 `status: complete`；完成结果再运行时只读回验证，不覆盖文件。

Wikipedia 逐篇写 JSONL，每 100 个来源行提交带累计字节数和 SHA256 的进度日志。中断后重新遍历固定来源至已提交边界，先验证已提交前缀，只丢弃该边界后未提交的尾部，再继续写入。这会重复读取一些源数据，但不重复接纳已提交文章。首次初始化与不能验证的损坏状态明确报错，不能静默从空文件重建。

Fed 请求复用原 `FedFetcher`：仅 HTTPS `www.federalreserve.gov`，串行每秒至多一次请求，每个 HTML 响应最多 2MB、有限超时和重试。每页先提交包含 URL、字节数、正文 SHA256 和 HTML 的原子缓存，再保留原 HTML 快照。中断后从校验过的页面缓存重建尚未完成的金融部分，已完成部分不会重抓。参数或源版本发生变化时必须选择新扩充版本，旧输出保留。

流式迭代无需把英文 Wikipedia 全量下载或转为内存列表，但 Parquet 解码与网络读取仍有缓冲，不能称为零内存。[Hugging Face Datasets 官方 streaming 说明](https://huggingface.co/docs/datasets/stream)

默认的 200MB + 12MB 是候选 JSONL 上限，不是 token 数，也不包含所有缓存、清洗文本和检查点。正式产物还需保留 HTML、恢复缓存、清洗结果和 `uint16` token 文件；后者净载荷为每 token 2 字节。实际占用与 token 数在构建后记录，不承诺单凭这些来源达到 100M 通用或 25M 独立金融 tokens。

## 进入训练前的条件

- 保持原 `seed_v1` 的 tokenizer 不变；新的候选集不能重新训练或替换词表。
- 排除原 seed 的 train 与 validation 文档身份、文本重复／近重复，以及冻结的任务评测文本；扩大 Wikipedia 前缀本身不能被算作全新文档数。
- 按文档划分新增 train/dev，保留原验证集用于可比评估，分别报告新增独立文档与实际 tokenizer token 数。
- 承接权重时记录原 300 步检查点的 hash 与新语料 provenance。新数据不能冒充同一数据流的精确 optimizer/cursor 续跑；后续恢复必须校验新阶段的独立运行合同。

## 验证记录

离线测试 `tests/test_expansion_sources.py` 11 项通过，覆盖：完成结果只读重用；Wikipedia 已提交前缀恢复、未提交尾部处理和前缀损坏拒绝；完成标记／原子改名之间的中断；真实链接白名单与发布日期；旧结构提取／版权标记排除；金融缓存恢复与损坏拒绝；参数改变拒绝；原 HTML 快照修改拒绝。

2026-10-11 执行小规模真实网络验证：通用上限 1MB、30 篇上限，金融年份 2011 与 2025、最多 2 篇、200KB 上限。金融按日期取前两篇，实际取到 2011 年的两个旧结构样例。所有候选文件留在被 Git 忽略的本地目录 `data/raw/expansion_network_probe_20261011/corpora`，未上传原文。

| 小规模候选 | 实际篇数 | 实际字节数 | SHA256 |
|---|---:|---:|---|
| 通用英文 | 28 | 980,278 | `0dc2e5955a33057ef735e99cdea8dcfebc19c2746f4102d3a0e1a9312fb270e6` |
| FOMC 金融英文 | 2 | 116,347 | `ecbc426ab4bbfec62f1ba03efe9bfde3e51037276406868216ee9bae8658516d` |

小规模目录包含恢复缓存等共 2,262,672 字节。金融无跳过；manifest `complete`。按完全相同参数再运行返回 `reused: true`。这只是获取、解析和重用验证，未据此主张达到正式语料规模、训练吞吐或模型效果。默认限量下载和准备的后续实际统计应记录在阶段报告中。

## 默认限量下载的实际结果

2026-10-11 执行默认 200MB Wikipedia 上限与 2011–2025 FOMC 来源获取。manifest 于 `2026-10-11T10:42:25.249898+00:00` 完成；本地文件位于 `data/raw/expansion_v1/corpora`，没有提交原文到 Git。

| 默认候选 | 实际篇数 | 实际字节数 | SHA256 |
|---|---:|---:|---|
| 通用英文 | 28,486 | 199,996,572 | `5b2b8def467868f177afd438c27ba4254fe41d746a5bbabb4bbbf3046769d547` |
| FOMC 金融英文 | 120 | 6,977,681 | `5a9b357c4b979b5e85dc7622e28c5793068baeb67a48abbd218e81e7d8219a0a` |

目录含来源 HTML、恢复缓存和元数据共 247,243,919 字节；这是输出目录大小，未包含外部 HF 缓冲／缓存或准备产物，也不是进程峰值内存。金融保存 12 个发现页面快照，120 篇均通过已审查的新旧正文结构与版权标记检查，`skipped` 为空。

这些是来源候选数，包含旧 seed 重叠，不可写作新增独立文档数。新篇数、分区和 tokenizer token 数由后续构建报告给出；本记录没有启动新阶段 Colab 训练。
