# 下一步：审核目标资产与立场标签

**当前交接：**用户已交付2000条CSV并明确确认1993条标题标签，7条待查；[接收说明](POST_DOMAIN_NEXT.md)记录核验及进入SFT前的关卡。下面的0条人工确认是最初生成时记录。

2026-10-30后续更新：按用户要求，已生成包含原200条的2000条不同标题AI初稿和离线审核工作台。每条带建议立场、原文证据、中文理由、历史资产候选及优先级；分20批，每批100条。人工确认均保持空白。首选工作台及快捷键说明见[审核工作台](REVIEW_WORKSPACE.md)，数量及证据边界见[2000条准备记录](../reports/task_review_2000_20261030.json)。实际材料位于本地`runs/review_packages/review_2000_20261030/`；原200条文件保持不变。以下保留首批200条导出说明。

2026-10-30。本机已生成首批200条审核材料，保留原来的AI初判和理由，并补充原作者实体表的历史代码候选。实际材料存放在忽略目录`runs/review_packages/review_200_20261030/`，没有将新闻原文或逐行标签提交到GitHub。

本地文件：

- `review_200_to_complete.csv`：用Excel或其他支持UTF-8的表格工具打开并逐行审核。
- `READ_ME_FIRST.md`：审核规则与分组要求。
- `manifest.json`：生成时的输入与产物哈希、计数及证据边界。

汇总见[准备记录](../reports/task_review_readiness_20261030.json)。200行中有116行能精确匹配到唯一的**来源代码候选**，2行存在多个来源代码候选，82行未匹配。这不是116个已核实的当前证券标识；原实体表涵盖历史名称及多种目标，不能全部当作美股。

## 需要填写的字段

| 字段 | 人工操作 |
|---|---|
| `asset_identifier` | 核对指定目标的资产身份、市场与代码后填写，不能直接照抄历史来源代码 |
| `human_stance` | 阅读原文对指定目标的投资方向，填写`bullish`、`bearish`、`neutral`或`exclude` |
| `human_reviewed` | 只有实际审核该行后才改为`true` |
| `notes` | 记录排除、条件性判断、资产映射依据或仍有的歧义 |
| `event_group_id` | 将确定属于同一事件的新闻赋予共同组号；未知则留空待查 |

`source_sentiment`是作者的实体情绪标签，`proposed_stance`和`ai_rationale`是AI草稿；均保留供比较，不能改名成人工真值。`source_symbol_candidates`、`source_symbol_match`只是精确别名查询结果，未核验当前上市状态、交易所或证券身份。

最初生成时人工确认立场和已核实资产标识均为**0**；当前实际数量等待接收工作台导出后统计。本包没有创建正式SFT数据、DPO偏好对或封存测试集。200条是首批规则审核样本，也不是原计划规模的标签语料；一人审核不等于双人独立一致性验证。

只报告已经发生的价格或财报变化，不自动代表看多或看空。单纯提出问题而未给结论、监管机构、模糊目标或不支持的资产，需要审核或排除。多目标新闻应分别判断，条件性观点应保留条件。

## 生成与保留

在仓库根目录、现有材料均已准备好的情况下执行：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts/build_task_review_package.py --report reports/task_review_readiness_20261030.json
```

脚本仅读取本地文件，核验固定来源、原审核表和AI初判文件的哈希及逐行身份；不会下载数据、查询当前股票代码或批准标签。默认需要：

- `data/raw/sentfin_review_v1/review_200.csv`
- `data/raw/sentfin_review_v1/review_200_ai_proposals.csv`及其两个manifest
- `data/raw/candidates/sentfin/entity_list_comprehensive.csv`

生成后可直接继续编辑CSV。未改动的完整包重复运行时只复用，保持文件不变；已有手工修改或不完整文件时明确停止，不覆盖原材料。需要另一份初始草稿时指定新的`--output`目录，同时保留已审核文件。

审核完成后的导入与数据切分仍须校验标签、资产映射、来源权限和重复关系。同一来源新闻、同一文本的多目标、词法近重复及同一事件必须联合分组后划分train/dev/calibration/test。当前`source_headline_id`和`text_group_id`仅是分组草稿，不能取代事件与语义重复审核。
