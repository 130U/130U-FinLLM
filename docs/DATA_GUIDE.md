# 数据准备与审核

首先明确：任务是“给定文本对指定资产表达的投资立场”，不是整句话的普通正负情绪。历史行情陈述默认中性；不能用之后的涨跌补标签。

## 输入文件

以下文件由用户材料或经过来源核查的公开材料组成；仓库没有虚构已取得的正式语料。

```text
data/raw/
  task.jsonl                 # 目标条件化人工确认标签
  general_train.jsonl         # 通用训练文本
  finance_train.jsonl         # 金融训练文本
  general_validation.jsonl    # 预训练诊断保留文本
  finance_validation.jsonl    # 金融诊断保留文本
  tokenizer_train.jsonl       # 通用/金融训练文本的代表性小样本
```

每行一个 JSON 对象。正式原文需要 `text, source, rights_note, training_allowed:true`。数据标签需要 `id,target,label`，并建议 `created_at`（ISO8601 含时区）、`event_id`、人工审核信息。验证原文的 `split` 如有填写应为 `validation`，训练原文为 `train`。字段缺失不能编造日期或权利。

`training_allowed` 表示来源权利已经核查过，不是勾上就产生许可。第三方镜像或公开可见不自动意味着训练许可。先检查候选原始数据卡、原始来源和适用条件，再导入。X 内容尚未取得明确适用训练权限时，不作为首轮训练材料；可以使用有权使用的英文金融材料和独立自编技术样本。

标注字段的格式示例见 `data/examples/annotation_template.jsonl`，该文件只用于理解格式，不能当正式数据。

## 标签与目标

- `bullish`：针对目标表达买入、做多、看涨或正面投资前景。
- `bearish`：针对目标表达卖出、做空、看跌或负面投资前景。
- `neutral`：可核验地提及目标，但未表达方向。
- 反讽、引述归属或资产关系不清的样本先保留待审核，不硬塞中性。公开数据的 positive/negative 标签必须按上述含义审核。

同一帖子多 target 共享原始 `id`；同事件或重复来源提供相同 `event_id/group_id`。当前自动检查是规范化精确匹配＋64-bit SimHash（Hamming≤3），不保证检出所有语义改写；人工事件分组补充这一范围。

## 正确顺序

1. 先审核约200条规则样本；记录分歧；取得足以保留各类验证样本的数据。两名标注人独立复核与一人延时复标应分别报告。
2. 运行 `prepare-task`，按重复/事件组做约70/10/10/10切分。有可靠时间则按组最晚时间排序；否则明确标为非时间外推。校准分区再按组分成两个子集。
3. 将任务 held-out 原文和两类预训练验证原文合成 `frozen_exclusions.jsonl`。正式 tokenizer 和训练分片排除这些内容；预训练验证分片只排除任务 held-out 内容，避免把自己全部过滤掉。
4. 自训词表，检查实际8192个token；语料太少导致不足时增加训练材料，用新输出目录重做。不要扩大特殊 token 后沿用旧参数计数。
5. 分片编码，生成哈希清单；默认每片5M tokens约10MB。精确去重索引放磁盘SQLite，token用uint16保存，只有当前batch转int64。
6. 用 `mix-corpora` 精确生成20M金融＋5M通用tokens。输入不够时会循环已有源，清单单独记录源token数量与混合累计数量；重复回放不等于新增独立语料。

完整可执行命令放在 `01_prepare_data.ipynb`。分片目录已有清单或旧残片时，应选新的版本目录，不能覆盖正在被检查点引用的数据。

## 受控流式下载工具

`pip install -e '.[data]'` 后可以使用 `fetch-corpus`。只访问显式指定的数据集和固定revision，按文档条数/字节数停止；需先核查使用条件。初始默认上限是1000份文档、20MB，不会下载整个大型库。

```bash
python -m finllm fetch-corpus --dataset DATASET_ID --revision FULL_40_CHARACTER_COMMIT \
  --max-documents 1000 --max-bytes 20000000 --rights-note "经核查的具体条件" \
  --training-rights-reviewed --output data/raw/general_candidate.jsonl
```

`DATASET_ID` 和 revision 必须填写已核对的值；这不是已执行的下载记录。候选可从 [FineWeb-Edu 数据卡](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 开始核查，金融来源可核查 [SEC EDGAR 官方访问规则](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data)。本版本没有内置X抓取器或代替用户同意来源条款。

流式源中的单份超大文档会跳过并在后续版本需要时另行分段。原文中已有的否定词、cashtag、数字和百分比保留；不要做会删除它们的通用文本清洗。

## DPO 数据

```bash
python -m finllm make-preferences --input data/task/train.jsonl --output data/task/preferences_candidates.jsonl
```

该命令从已审核训练标签构造候选 chosen/rejected，默认 `preference_reviewed:false`。审核偏好对后将已确认的记录另存为 `preferences_train.jsonl` 并标记 true。正式训练拒绝未审核偏好对、非训练分区、相同答案或未知类别。不要从dev/cal/test构造偏好。

## 评估

开发集用于模型和训练参数选择；最终模型冻结后才分别拟合温度与复核阈值。仅最后对test报告结果，不回看test调参。真实效果至少报告macro-F1、每类表现、样本量、NLL/Brier/ECE和复核覆盖率。

```bash
python -m finllm baseline --train data/task/train.jsonl --input data/task/development.jsonl --output reports/private/baseline_dev.json
```

该命令包括多数类及目标感知TF-IDF线性基线。FinBERT比较、重复seed、事件组bootstrap和未见资产压力测试仍需实际数据与后续实验，不视为本次小实验已完成。
