# Colab 到 Kaggle：承接已有领域模型

这次迁移读取已完成的`domain_v1`领域检查点、用户确认的实体标签、8K词表及上游排除池，保持原Drive成果。Kaggle的SFT是新实验：只加载领域模型权重，重新建立优化器。截图的T4 ×2有两张GPU，本实现使用其中一张，不把两张显存视为一块。

## 1. 在 Colab CPU 导出

打开[04b Kaggle导出](https://colab.research.google.com/github/130U/130U-FinLLM/blob/main/notebooks/04b_kaggle_export.ipynb)，选择「运行时 → 更改运行时类型 → 无」，依次运行并挂载原Drive。程序先核验领域权重、词表、任务标签、四份真实预训练排除池及上游权重链，再流式打包。

记录输出中的ZIP SHA256和`manifest_sha256`；浏览器提示下载时，等待`130u_kaggle_transfer.zip`下载完成。文件可能超过1GB，以实际输出为准。包包含模型权重和原始标题，应保存在本人电脑并作为私人数据处理；当前没有独立读取用户Drive，实际云端校验结果以运行输出为准。

## 2. 准备 Kaggle 输入与 Notebook

在自己的Kaggle账户将转运ZIP创建为**Private**数据集，检查Notebook本身也是Private。在用户已有Notebook中，通过File → Import Notebook导入本项目`notebooks/04c_kaggle_sft.ipynb`，再通过Add Input添加刚才的私有数据集。保留Session options中GPU T4 ×2和Internet on。

在04c的配置单元粘贴04b输出的ZIP SHA256和manifest SHA256。Kaggle可能自动解压上传的ZIP：Notebook会识别解压目录并按manifest SHA、逐文件SHA核验；若ZIP保留则核验ZIP SHA。两个恢复SHA在首次运行时留空。

## 3. 运行与备份

按顺序运行，记录第一格完整40位代码SHA。程序核验领域→扩充→seed母模型链和实际排除池，单步验证后训练并在开发集比较。运行结果位于`/kaggle/working`；需下载最终私有恢复ZIP及其两个SHA，或确保私有Notebook版本保存了输出，不能把工作目录视为永久备份。

跨会话恢复要附加原转运数据和私有恢复数据，填入恢复清单或ZIP SHA，并将第一格REF固定为首次运行的完整代码SHA。严格恢复还要求路径、数据、设备、AMP和PyTorch/CUDA等运行身份匹配；不符合时停止而非假装精确续训。突然中断时未另行保存的本地进度可能丢失。

项目输出是新闻标题的字面实体立场试验；未验证证券身份、完整事件及时间隔离，不能代表X场景或投资收益。Kaggle与Drive的实际用户环境尚待执行验收。

本地验证：31项转运、四池核查、SFT及中断恢复联合测试通过；另校验17本Notebook格式与Python语法。测试覆盖模拟Kaggle自动解压、微型seed→领域训练→导出→接收→SFT单步及恢复，但不等于真实用户云端成功。
