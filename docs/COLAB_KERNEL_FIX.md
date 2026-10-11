# Colab 内核安装与训练入口修复

问题记录：[GitHub #1](https://github.com/130U/130U-FinLLM/issues/1)。2026-10-09 用户截图显示，02a 已选择 Tesla T4，但训练单元格在导入 `finllm.runtime` 时抛出 `ModuleNotFoundError`。同一组截图的 01a 显示 CPU 运行时；它的断开连接提示无法单独证明内存不足或确定断开原因。

2026-10-11 更新：用户提供的 T4 截图显示 02a 已完成 61 次更新、999,424 tokens，并成功备份到 Drive；随后 02b 日志及备份输出显示累计 300 步、4,914,941 有效 tokens，最终快照校验通过。此前的导入故障已在用户运行中越过。实际报告数字及证据边界见 [T4 训练记录](T4_PILOT.md)；这是用户提供的运行证据，未独立读取云端检查点。

## 根因与修复

项目采用 `src` 布局。Notebook 的 Python 内核早于 `pip install -e .` 启动；editable 安装产生的 `.pth` 会被后续新 Python 进程读取，已运行内核不会自动重新读取。因此原先 `python -m finllm environment` 成功，不能证明后面的内核内 `from finllm.runtime import ...` 会成功。

公共安装单元格现在显式激活当前仓库的 `src`，刷新仅属于 `finllm` 的缓存模块，并分别检查当前内核、新 Python 进程的实际导入路径和检查点恢复接口。Torch 不会被重新加载。

旧项目安装信息仍限制 `tokenizers<0.23` 时，先升级 tokenizers、后更新项目会产生中间依赖冲突。现在把 editable 项目和兼容的 tokenizer 放进同一次 pip 事务，约束为当前已安装的完整 Torch 版本；完成后检查项目基础依赖及 Transformers 的 tokenizer 约束。

同时修复两处训练流程问题：

- 达到训练预算后重复运行会只读复用原结果，保留 checkpoint、备份和原始吞吐／显存记录。仅从检查点恢复且缺少原汇总时，不编造新的吞吐或显存成绩。
- 训练失败后，02a 的最终备份单元格明确停止，避免继续显示完成；训练正常进行时逐次显示更新步数和 loss。

## 验证方式与边界

新增 `tests/test_notebook_execution.py`，使用真正持续运行的 Jupyter 内核，先启动内核再实际执行 editable 安装。测试重现旧导入失败，并执行生成 Notebook 中的安装、01a 配置恢复、02a 恢复、配置、训练、备份及重复运行单元格。另启一个独立 Python 环境和内核，在不同目录恢复数据及 checkpoint，验证从 2 次更新继续到 3 次更新，且持久目录与工作目录的 checkpoint 哈希一致。

测试中的外部 Git 和 Drive 使用本地目录，依赖安装离线复用已有库，GPU 预算明确替换为微型 CPU 测试配置。Python 包安装、内核导入、模型参数更新和 checkpoint 恢复实际执行。这是 Notebook 运行链路验证，不是云端 Google Drive 服务或 T4 性能验收。此前的测试只验证了子进程和语法，没有覆盖这个内核生命周期差异。

本地完整测试 120 项通过（73.26 秒），9 本 Notebook 格式／语法通过，`pip check` 通过。机器记录见 [colab_kernel_fix.json](../reports/colab_kernel_fix.json)。CI 在 Python 3.12、3.13 下执行同一完整测试集，结果以该修复提交的 Actions 为准。

## 用户继续入口

已有 01a 的 `seed_v1` Drive 备份时，直接从 [最新版 02a](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/02a_seed_pilot.ipynb) 重新打开并保存副本。保持 T4、`SEED_VERSION = "seed_v1"`，从第一格顺序运行。安装格应显示“当前 Notebook 内核导入检查通过”和“新 Python 进程导入检查通过”；后续恢复已有数据、词表及进度。

无需删除 Drive 中的文件或重做 01a。旧版已经打开或保存在 Drive 的 Notebook 不会自动更新单元格代码，应使用上面的新版入口。若训练完成后再次运行，返回 `resumed_no_training: true`；原 `run_summary.json` 保留该次真正训练的测量结果。
