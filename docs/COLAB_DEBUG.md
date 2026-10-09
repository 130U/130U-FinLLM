# 01a 重复运行与依赖冲突修复

2026-10-09。用户提供的 Colab PDF 显示代码版本 `1c265b3`，在再次执行构建单元格时抛出 `FileExistsError: Use a new version: data/raw/seed_v1/prepared`。下载和 Drive 备份在此前已显示成功。目录存在本身不能判定成果损坏，旧入口把完成后的重复运行也拒绝了。

另一个独立问题是依赖约束：该 Colab 自带 Transformers 5.18.0，要求 tokenizers >=0.23.1,<0.24，旧入口却安装 <0.23。项目现在支持 <0.24，安装时同时遵守现有 Transformers 的约束，安装后再检查；保留 Colab 的 PyTorch。Hugging Face 未登录提示不是此次中断原因，日志已经显示下载成功。

## 已修复行为

- 已完成的下载、任务排除池、词表和分片通过清单、来源、配置及哈希校验后只读复用，兼容旧版生成的成果。
- 不完整、损坏或参数变化时保留旧文件，明确提示恢复完整备份或选择新版本；不自动删除或覆盖。
- 新运行时可从 `MyDrive/130U-FinLLM/seeds/seed_v1/` 恢复原始数据、分片和词表。人工审核 CSV 的编辑不会被重新导出覆盖。
- 01a 备份前重新验证全部训练输入，避免把中断产物当成完成结果。
- 01a 可用 CPU；02a 才要求 GPU。用户本次日志是 `torch=2.11.0+cpu`、`cuda_available=false`，不能作为 T4 运行记录。

## 接着运行

1. 从 [GitHub 最新 01a](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/01a_seed_data.ipynb) 重新打开并保存一份 Notebook。旧 Drive 副本中的安装单元格不会随 GitHub 自动更新。
2. 保留 `SEED_VERSION = "seed_v1"`，从上到下运行。新运行时会恢复既有 Drive 快照；同一运行时会校验并复用本地成果。无需主动删除目录或重新训练词表。
3. 查看 `Existing seed verified`、真实 `seed_summary.json` 和备份完成信息；若完整性校验失败，按具体缺失文件定位，不强行跳过。
4. 完成后打开 [02a](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb)，选择 T4 GPU，确认 `cuda_available=true` 后再运行训练。

## 实际验证

旧代码在本地真实 seed 数据上复现同样的目录存在错误；修复后构建和只读校验均成功，69 个既有文件的 SHA256 和修改时间不变。机器记录见 [colab_rerun_fix.json](../reports/colab_rerun_fix.json)。

本地完整测试 112 项通过；9 本 Notebook 的格式与语法检查通过。隔离安装 tokenizers 0.23.1 后，词表、数据和训练相关 25 项测试通过，旧 8K 词表成功读取。新增 CI 覆盖 Python 3.12 和 3.13；CI 结果以对应提交的 Actions 为准。这些检查没有代替用户 Colab 的重新执行，也没有启动新的 GPU 预训练。
