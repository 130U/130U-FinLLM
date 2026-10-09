# 实现与验证状态

更新日期：2026-10-09。

最新修复：用户已在 Colab 执行 01a，提供的 PDF 显示下载和备份成功，之后重复执行构建单元格遇到目录已存在错误；同时出现 tokenizers/Transformers 约束冲突。两处已修复，见 `docs/COLAB_DEBUG.md`。用户端成果的完整性需由新版入口校验，尚未代用户重新运行。

本轮新增：用户提供的实际 Colab 截图显示 T4 小实验运行结束、`Mounted at /content/drive` 与备份完成。首批真实数据已在本地取得并处理：通用训练 4,976,894 tokens、金融训练 341,309 tokens，8192-token 自训词表，完整候选任务池在训练前排除。详细证据见 `reports/seed_data_preparation.json`、`reports/seed_model_input_check.json` 与 `docs/SEED_DATA.md`。24M 的约 1M-token T4 预训练仍未执行。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 本地测试集 | 112项通过，33.90秒 | Windows、Python3.12.14、CPU PyTorch2.14.1；含重复运行、数据完整性及依赖回归测试，不是T4验证 |
| Notebook | 原7本及新增01a/02a均做nbformat与代码语法检查，仓库输出为空 | 用户已运行00和01a；修复版01a及24M训练尚未代用户执行 |
| 小模型完整链 | 自训tokenizer；预训练/domain/SFT/DPO各2次更新；DPO恢复到第3次；校准/评测/预测成功 | 126,912参数、441token词表、自编合成样本；不能评价金融能力 |
| 正式24M架构 | 24,255,744参数；CPU上一次前向、反向和AdamW更新成功 | batch1、长度16、随机token；不是正式语料预训练 |
| 恢复一致性 | CPU/dropout场景，模型、优化器、调度器、RNG和下一批与连续运行一致 | 精确恢复绑定版本、设备类型与数据配置；没有跨硬件逐位保证 |
| 断点/备份失败 | 损坏/不完整检查点被拒绝；模拟中途保存失败可恢复上一完整版本；训练中断前的检查点已定期备份 | 本地目录模拟持久盘；Drive实际挂载仍需用户运行 |
| 依赖 | `pip check`通过 | 本地安装清单见reports；Colab保留其CUDA PyTorch并另记实际版本 |

机器可读证据：`reports/cpu_smoke_check.json`、`reports/full_config_cpu_check.json`、`reports/environment-local-cpu.txt`。这些文件只记录工程验证，没有真实金融测试分数。

## 已实现的功能

Dense/GQA/RoPE/QK-RMSNorm模型；可逆byte-level BPE；分组切分和词法近重复过滤；训练/评测隔离；uint16分片与mmap；精确80/20 token混合；全参数AdamW训练；FP16 AMP路径；标签位置SFT与DPO；完整恢复与定期检查点备份；目标输入规则；温度校准和独立阈值选择；JSONL预测；多数类和目标感知TF-IDF基线。

CLI 的流式入口及首批来源下载脚本已执行：Wikipedia 固定版本子集、美联储原创会议纪要、作者发布的 SEntFiN 候选池。来源条款、版本和哈希已记录；没有抓取 X 或下载整个大型语料库。

## 尚未完成

- 24M配置在用户Colab上T4实跑、峰值显存及稳定吞吐测量；00微型链已由截图确认。
- 扩充首批语料、目标级人工标注与资产映射、语义/事件级重复审查。
- 100M＋25M预算训练；首批8192词表已生成，后续更换输入版本需另建词表并记录检查点兼容性。
- 真实SFT/DPO、同新增标注的SFT对照、FinBERT比较、预训练消融、重复seed与事件组bootstrap。
- 对最终模型的独立校准、封存测试和实际使用验收。

下一步用最新版01a校验复用Colab成果，再运行02a。不要把数据准备、测试通过、合成样本loss下降或一次前向反向写成“金融模型已训练完成”。
