# 24M Trial｜Kaggle 轻量承接已有模型

范围：24M Trial 公开工程文档，记录已公开的 04d/04e 迁移与恢复路径。该路径后续已产生初始及平衡 SFT 结果，见[Trial 实验报告](experiment-results.md)；项目整体见[总览](../README.md)。

这条路径保留已训练的 `domain_v1` 模型权重和 8K 词表，不需要下载 1.54 GiB 的完整转运包。轻量包包含模型权重、词表、用户已审核的 1,008 条标题样本及身份清单，实际大小以输出为准，约 100 MB。Colab 中原来的模型和数据不会被删除。

## 1. Colab CPU 导出轻量包

打开[04d 轻量导出 Notebook](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/04d_colab_lite_export.ipynb)。选择「运行时 → 更改运行时类型 → 无」，从第一格顺序运行并授权已有 Drive。程序校验领域检查点、词表和任务，导出 `130u_kaggle_lite.zip`；记录输出的 `manifest_sha256` 和 ZIP SHA256，等待下载完成。

用户之前提供的 04b 输出显示四份实际预训练排除池各覆盖 1,008 条标题，且均 `missing: 0`。轻量包记录这份已报告的证据，不包含四份历史排除池；Kaggle 不能再次独立核验被省略的原始数据。

## 2. 在 Kaggle 导入训练页面与私有数据

在用户自己的 Kaggle 账号中把轻量 ZIP 创建为 **Private 数据集**，确认 Notebook 本身也是 Private。打开已有的 Kaggle Notebook，使用 File → Import Notebook 导入本项目 `notebooks/04e_kaggle_lite_sft.ipynb`，然后用 Add Input 添加该私有数据集。

选择 T4 x2、开启 Internet；当前实现只使用第一张 T4。填写 04d 输出的 `manifest_sha256`。如果 Kaggle 保留的是 ZIP 而非自动解压后的文件，还需填写 ZIP SHA256。其他恢复配置首次留空。

## 3. 训练、结果与保存

按顺序运行。第一次只加载 74 步领域模型的权重、创建新的 SFT 优化器；不是恢复 Colab 的优化器状态。程序在开发集比较 SFT 前后结果，输出恢复检查点及标记的 SHA256。

`/kaggle/working` 不保证永久保留：训练结束后保存 Private Notebook 版本并确认实际输出已保存，或者下载 `latest.pt` 与 `latest.pt.json`，同时保留私有轻量数据集、首次代码提交 SHA 和词表。跨会话恢复前须恢复原运行目录，并使用一致的代码、数据、路径、设备、PyTorch/CUDA 与 AMP 配置。

本页对应新闻标题中字面实体的探索性立场试验，资产身份、语义事件、时间隔离和 X 场景尚未完成验证；不代表证券价格预测能力。轻量导出与 Kaggle 输入、训练及恢复先经本地微型测试，后续用户已在实际 Kaggle 环境完成初始 SFT 与平衡 SFT；具体结果及各次备份证据范围见[Trial 实验报告](experiment-results.md)。
