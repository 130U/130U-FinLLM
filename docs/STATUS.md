# 实现与验证状态

更新日期：2026-10-11。

02a 的当前内核导入与依赖安装修复见 `docs/COLAB_KERNEL_FIX.md` 和 Issue #1。2026-10-11 用户提供的 Colab T4 截图显示，该入口已完成 61 次更新、999,424 tokens，Drive 备份校验通过。此前 01a 的目录复用修复记录见 `docs/COLAB_DEBUG.md`。

本轮新增：24,255,744 参数模型的 T4 试跑耗时 296.43 秒、吞吐 3371.56 tokens/秒，最后训练 loss 7.1595、限量验证 loss 7.2726，PyTorch 分配峰值显存约 1.058 GiB。数字由用户截图转录，未独立下载检查点复核；详见 `docs/T4_PILOT.md` 和 `reports/colab_t4_pilot_20261011.json`。准备 02b 保留试跑报告、验证恢复并继续到累计 300 步；02b 的云端运行尚未完成。

首批数据为通用训练 4,976,894 tokens、金融训练 341,309 tokens，8192-token 自训词表，完整候选任务池在训练前排除。数据证据见 `reports/seed_data_preparation.json`、`reports/seed_model_input_check.json` 与 `docs/SEED_DATA.md`。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 用户 T4 正式架构试跑 | 61/61 次更新，999,424 tokens；Drive 备份显示校验通过 | 用户截图证据；非代理独立复跑，云端恢复续训尚待验证；没有金融任务准确率 |
| 本地测试集 | 128项通过，127.50秒；其中8项续训测试 | Windows、Python3.12.14、CPU PyTorch2.14.1；包含真实持续运行Jupyter内核的安装、训练、备份、重复运行及新内核恢复 |
| Notebook | 10本格式与语法检查通过；01a/02a/02b 相关单元格实际执行 | Git/Drive替换为本地目录、离线复用依赖、微型CPU配置；02b 双内核恢复2→3→5及中断备份恢复通过，不等于该新入口已经在云端运行 |
| 小模型完整链 | 自训tokenizer；预训练/domain/SFT/DPO各2次更新；DPO恢复到第3次；校准/评测/预测成功 | 126,912参数、441token词表、自编合成样本；不能评价金融能力 |
| 正式24M架构 | 24,255,744参数；CPU上一次前向、反向和AdamW更新成功 | batch1、长度16、随机token；不是正式语料预训练 |
| 恢复一致性 | CPU/dropout场景，模型、优化器、调度器、RNG和下一批与连续运行一致 | 精确恢复绑定版本、设备类型与数据配置；没有跨硬件逐位保证 |
| 断点/备份失败 | 损坏/不完整检查点被拒绝；模拟中途保存失败可恢复上一完整版本；训练中断前的检查点已定期备份 | 本地目录模拟持久盘；Drive实际挂载仍需用户运行 |
| 依赖 | `pip check`通过 | 本地安装清单见reports；Colab保留其CUDA PyTorch并另记实际版本 |

机器可读证据：`reports/cpu_smoke_check.json`、`reports/full_config_cpu_check.json`、`reports/environment-local-cpu.txt`、`reports/seed_continuation_checks.json`。这些文件只记录工程验证，没有真实金融测试分数。

## 已实现的功能

Dense/GQA/RoPE/QK-RMSNorm模型；可逆byte-level BPE；分组切分和词法近重复过滤；训练/评测隔离；uint16分片与mmap；精确80/20 token混合；全参数AdamW训练；FP16 AMP路径；标签位置SFT与DPO；完整恢复与定期检查点备份；目标输入规则；温度校准和独立阈值选择；JSONL预测；多数类和目标感知TF-IDF基线。

CLI 的流式入口及首批来源下载脚本已执行：Wikipedia 固定版本子集、美联储原创会议纪要、作者发布的 SEntFiN 候选池。来源条款、版本和哈希已记录；没有抓取 X 或下载整个大型语料库。

## 尚未完成

- 02b 在用户 T4 上的真实恢复和累计 300 步续训；长时间吞吐稳定性仍未测量。
- 扩充首批语料、目标级人工标注与资产映射、语义/事件级重复审查。
- 100M＋25M预算训练；首批8192词表已生成。后续语料扩充优先复用词表，需显式记录数据变更与检查点兼容策略；不能把不同数据伪装为精确恢复。
- 真实SFT/DPO、同新增标注的SFT对照、FinBERT比较、预训练消融、重复seed与事件组bootstrap。
- 对最终模型的独立校准、封存测试和实际使用验收。

下一步打开 02b，复用现有 seed_v1 数据、词表和检查点。不要直接打开使用其他默认路径的 03，也不要将这次 1M-token 试跑写成“金融模型已训练完成”。
