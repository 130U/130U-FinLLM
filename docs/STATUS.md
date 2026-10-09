# 实现与验证状态

更新日期：2026-10-09。

最新修复：02a 在已运行的 Notebook 内核找不到刚通过 editable 安装的 finllm，而子进程可以导入。共享安装入口现激活当前内核并双重核对导入路径；依赖在一次安装事务内更新。另修复完成预算后重复运行清空指标，以及失败后继续最终备份的问题。真实 Jupyter 内核集成测试已执行，见 `docs/COLAB_KERNEL_FIX.md` 和 Issue #1；云端 T4 本次尚未重跑。此前 01a 的目录复用修复记录仍见 `docs/COLAB_DEBUG.md`。

本轮新增：用户提供的实际 Colab 截图显示 T4 小实验运行结束、`Mounted at /content/drive` 与备份完成。首批真实数据已在本地取得并处理：通用训练 4,976,894 tokens、金融训练 341,309 tokens，8192-token 自训词表，完整候选任务池在训练前排除。详细证据见 `reports/seed_data_preparation.json`、`reports/seed_model_input_check.json` 与 `docs/SEED_DATA.md`。24M 的约 1M-token T4 预训练仍未执行。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 本地测试集 | 120项通过，73.26秒 | Windows、Python3.12.14、CPU PyTorch2.14.1；包含真实持续运行Jupyter内核的安装、训练、备份、重复运行及新内核恢复 |
| Notebook | 9本格式与语法检查通过；01a配置恢复及02a训练恢复单元格实际执行 | Git/Drive替换为本地目录、离线复用依赖、微型CPU配置；不等于Google Drive云服务或T4验收 |
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

下一步直接打开最新版02a，复用现有seed_v1备份；无需重做01a。不要把数据准备、测试通过、合成样本loss下降或一次前向反向写成“金融模型已训练完成”。
