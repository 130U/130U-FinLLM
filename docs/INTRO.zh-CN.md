# 130U-FinLLM｜简短中文介绍

130U-FinLLM 是面向个人金融研究的语言模型项目，由 **2.4B 自用版本**与 **24M 公开 Trial** 组成。本仓库用英文展示模型架构、训练体系与工程复盘，并公开 Trial 的代码、配置、Notebook 和实验记录。

项目从原创金融观点与人工审核文本出发，围绕目标、观点持有人、证据范围和判断条件组织监督。两项核心设计分别是：金融语义许可，区分哪些判断应随语义变化、哪些应当保持；保持训练目标的扩容工程，将样本家族、损失计量、多卡计算与恢复语义连接起来。

24M 用于打通完整训练流程并形成可复用的工程经验，2.4B 面向个人研究使用。两个版本各自管理词表、权重和校准参数，实验结果保留明确的版本归属。

[English README](../README.md) · [Project overview](PROJECT_OVERVIEW.md) · [Training system](TRAINING_SYSTEM.md) · [Run the Trial](TRIAL_GUIDE.md)
