# 实现与验证状态

更新日期：2026-10-11。

02a 的当前内核导入与依赖安装修复见 `docs/COLAB_KERNEL_FIX.md` 和 Issue #1。2026-10-11 用户提供的 Colab T4 截图显示，该入口已完成 61 次更新、999,424 tokens，Drive 备份校验通过。此前 01a 的目录复用修复记录见 `docs/COLAB_DEBUG.md`。

本轮验收：用户提供的 02b 日志显示，24,255,744 参数模型已从第 62 步续训到累计 300 步、4,914,941 个有效 tokens，最后训练 loss 5.4151、限量验证 loss 5.5419。本段 238 次更新耗时 1150.17 秒、吞吐 3390.06 tokens/秒、PyTorch 分配峰值显存约 1.063 GiB。用户随后提供最终快照输出：7 个文件、615,936,768 字节、`verified: true`。续训与最终备份完成；代理未独立下载检查点或读回 Drive 清单。详见 `docs/T4_PILOT.md`、`reports/colab_t4_pilot_20261011.json` 和 `reports/colab_t4_continuation_20261011.json`。

首批数据为通用训练 4,976,894 tokens、金融训练 341,309 tokens，8192-token 自训词表，完整候选任务池在训练前排除。数据证据见 `reports/seed_data_preparation.json`、`reports/seed_model_input_check.json` 与 `docs/SEED_DATA.md`。

新增 `01b_expand_data.ipynb` 和 `02c_expanded_pretrain.ipynb`：建立独立扩充版本，复用原词表、排除原 seed 与冻结任务文本，显式冻结母检查点并仅承接模型权重。新阶段重新建立优化器／调度器／数据游标，之后同版本只能严格恢复。默认先验证第 1 步，再至新阶段 610 步，并比较原 seed 验证分片的 before/after loss；尚无本阶段 Colab/T4 运行成绩。操作与验收见 `docs/DATA_EXPANSION.md`。

本地默认来源下载及构建已完成：新增通用 train 为 26,267 篇、41,499,364 tokens，金融 train 为 84 篇、972,438 tokens；新增 validation 分别为 1,394 篇／2,235,968 tokens 和 4 篇／50,999 tokens。排除旧 seed 825＋32 篇，并检查 10,686 条冻结任务原文；原词表和 seed 清单未改变。1,215,545-token 金融混合分片是上述源的重排，不计为再次新增。详见 [本地扩充报告](../reports/expansion_local_validation_20261011.json)。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 用户 T4 试跑与续训 | 累计 300/300 次更新、4,914,941 有效 tokens；验证 loss 7.2726→5.5419；最终快照输出校验通过 | 用户截图、日志及备份输出证据；非代理独立复跑或 Drive 读回；不能据此证明新虚拟机恢复；没有金融任务准确率 |
| 本地测试集 | 205项通过，274.58秒 | Windows、Python3.12.14、CPU PyTorch2.14.1；包含真实持续运行Jupyter内核的安装、训练、备份、重复运行及新内核恢复 |
| Notebook 工程检查 | 12本格式、语法与无存储输出检查通过；新增01b/02c代码单元格实际执行 | 01b在持续运行Jupyter内核中执行一次data-extra安装、真实8KBPE数据构建／备份／重复复用；02c在两个新内核中完成微型正式CPU权重承接、单步验证、训练、原验证集前后测量、备份和新目录恢复。Git/Drive/外部下载使用本地替代；不是新阶段T4训练 |
| 扩充数据实际构建 | 544.91秒；采样RSS峰值344.7MiB；候选、清洗及token产物合计560.2MiB | 本地CPU、0.1秒采样；不含旧seed、检查点及外部缓存；不是精确分配峰值或Colab/T4显存测量 |
| 小模型完整链 | 自训tokenizer；预训练/domain/SFT/DPO各2次更新；DPO恢复到第3次；校准/评测/预测成功 | 126,912参数、441token词表、自编合成样本；不能评价金融能力 |
| 正式24M架构 | 24,255,744参数；CPU上一次前向、反向和AdamW更新成功 | batch1、长度16、随机token；不是正式语料预训练 |
| 恢复一致性 | CPU/dropout场景，模型、优化器、调度器、RNG和下一批与连续运行一致 | 精确恢复绑定版本、设备类型与数据配置；没有跨硬件逐位保证 |
| 断点/备份失败 | 损坏/不完整检查点被拒绝；模拟中途保存失败可恢复上一完整版本；训练中断前的检查点已定期备份 | 故障场景为本地目录模拟；用户另提供真实 Drive 最终备份成功输出，未独立核对云端文件 |
| 依赖 | `pip check`通过 | 本地安装清单见reports；Colab保留其CUDA PyTorch并另记实际版本 |

机器可读证据：`reports/cpu_smoke_check.json`、`reports/full_config_cpu_check.json`、`reports/environment-local-cpu.txt`、`reports/seed_continuation_checks.json` 记录工程检查；[扩充报告](../reports/expansion_local_validation_20261011.json) 记录本地实际下载、构建、资源测量和本轮工程检查；`reports/colab_t4_continuation_20261011.json` 记录用户提供的真实语料预训练及备份输出。均没有金融立场测试分数。

## 已实现的功能

Dense/GQA/RoPE/QK-RMSNorm模型；可逆byte-level BPE；分组切分和词法近重复过滤；训练/评测隔离；uint16分片与mmap；精确80/20 token混合；全参数AdamW训练；FP16 AMP路径；标签位置SFT与DPO；完整恢复与定期检查点备份；目标输入规则；温度校准和独立阈值选择；JSONL预测；多数类和目标感知TF-IDF基线。

CLI 的流式入口及首批来源下载脚本已执行：Wikipedia 固定版本子集、美联储原创会议纪要、作者发布的 SEntFiN 候选池。来源条款、版本和哈希已记录；没有抓取 X 或下载整个大型语料库。

## 尚未完成

- 用户 T4 的新虚拟机恢复、跨多次会话运行及超过本段约 19 分钟的吞吐稳定性仍需验证。
- 完整预算所需的更广语料、目标级人工标注与资产映射、语义/事件级重复审查。
- 100M＋25M预算训练；首批8192词表已生成。后续语料扩充优先复用词表，需显式记录数据变更与检查点兼容策略；不能把不同数据伪装为精确恢复。
- 真实SFT/DPO、同新增标注的SFT对照、FinBERT比较、预训练消融、重复seed与事件组bootstrap。
- 对最终模型的独立校准、封存测试和实际使用验收。

下一运行路径为 01b 扩充准备／备份 → 02c 新阶段单步验证／610 步训练／原验证集对比／最终备份。02b 当前 300 步成果无需重训；不要直接打开使用其他默认路径的 03，也不要将现有约 4.91M-token 通用预训练或新入口交付写成“金融模型已训练完成”。
