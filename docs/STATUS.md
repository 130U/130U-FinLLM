# 实现与验证状态

更新日期：2026-10-30。

02a 的当前内核导入与依赖安装修复见 `docs/COLAB_KERNEL_FIX.md` 和 Issue #1。2026-10-11 用户提供的 Colab T4 截图显示，该入口已完成 61 次更新、999,424 tokens，Drive 备份校验通过。此前 01a 的目录复用修复记录见 `docs/COLAB_DEBUG.md`。

本轮验收：用户提供的 02b 日志显示，24,255,744 参数模型已从第 62 步续训到累计 300 步、4,914,941 个有效 tokens，最后训练 loss 5.4151、限量验证 loss 5.5419。本段 238 次更新耗时 1150.17 秒、吞吐 3390.06 tokens/秒、PyTorch 分配峰值显存约 1.063 GiB。用户随后提供最终快照输出：7 个文件、615,936,768 字节、`verified: true`。续训与最终备份完成；代理未独立下载检查点或读回 Drive 清单。详见 `docs/T4_PILOT.md`、`reports/colab_t4_pilot_20261011.json` 和 `reports/colab_t4_continuation_20261011.json`。

首批数据为通用训练 4,976,894 tokens、金融训练 341,309 tokens，8192-token 自训词表，完整候选任务池在训练前排除。数据证据见 `reports/seed_data_preparation.json`、`reports/seed_model_input_check.json` 与 `docs/SEED_DATA.md`。

`01b_expand_data.ipynb` 和 `02c_expanded_pretrain.ipynb` 建立独立扩充版本，复用原词表、排除原 seed 与冻结任务文本，冻结母检查点并仅承接模型权重。用户已完成这两个入口：扩充阶段 610 步、9,994,240 tokens，最后训练 loss 4.1933、新通用验证子集 loss 4.1311，原 seed 保留子集训练后 loss 4.4032；1→610 段耗时 2700.80 秒、3694.41 tokens/秒、分配峰值显存约 1.063 GiB。最终 7 文件／615,996,721 字节快照输出为 `verified: true`。这是用户日志和截图证据，未独立读回 Drive；缺少 before 数值，不能量化本轮保留集变化。见 `reports/colab_t4_expansion_20261011.json`。

`02d_expanded_continue.ipynb` 已由用户完成。2026-10-30 从用户保存的 GitHub 提交 `e14b85e` 独立读回输出：扩充阶段累计1220步、19,988,195个有效tokens；训练loss 3.6796、新通用验证loss 3.6929，原seed保留子集before/after为4.4032→3.9842。最终备份输出为7文件、616,109,865字节、`verified: true`。含seed的模型路径累计24,903,136个有效tokens，不是唯一语料大小；尚未独立下载Drive权重验真。原执行Notebook保留于该历史提交，并按原始字节本地归档；可运行入口恢复为无输出的生成器版本，修复Colab保存输出导致的CI格式失败。见[读回记录](../reports/colab_t4_expanded_continuation_20261030.json)。

`03a_domain_pretrain.ipynb`已由用户完成。2026-10-30收到完整Colab输出：核验并冻结1220步母检查点后，独立`domain_v1`完成74步／1,211,960个有效tokens，金融验证loss为3.2375；同一seed通用保留子集为3.9842→4.0574。最终7文件／615,896,468字节备份输出为`verified: true`。含seed与扩充阶段的模型路径累计26,115,096个有效tokens，不是独立语料大小；本段没有完成原25M金融预算。见[领域记录](../reports/colab_t4_domain_20261030.json)和[下一步](POST_DOMAIN_NEXT.md)。证据来自用户粘贴日志，代理未独立下载Drive权重。

本地默认来源下载及构建已完成：新增通用 train 为 26,267 篇、41,499,364 tokens，金融 train 为 84 篇、972,438 tokens；新增 validation 分别为 1,394 篇／2,235,968 tokens 和 4 篇／50,999 tokens。排除旧 seed 825＋32 篇，并检查 10,686 条冻结任务原文；原词表和 seed 清单未改变。1,215,545-token 金融混合分片是上述源的重排，不计为再次新增。详见 [本地扩充报告](../reports/expansion_local_validation_20261011.json)。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 用户 T4 试跑与续训 | 累计 300/300 次更新、4,914,941 有效 tokens；验证 loss 7.2726→5.5419；最终快照输出校验通过 | 用户截图、日志及备份输出证据；非代理独立复跑或 Drive 读回；不能据此证明新虚拟机恢复；没有金融任务准确率 |
| 用户 T4 扩充阶段 | 610 步／9,994,240 tokens，新验证 loss 4.1311，原保留集 after loss 4.4032；最终快照输出校验通过 | 用户日志和截图；新增验证集与原保留集不直接横比；未独立读回 Drive、未提供 before 数值，没有金融任务准确率 |
| 用户 T4 扩充续训 | 累计1220步／19,988,195有效tokens；新验证loss 3.6929、seed保留集4.4032→3.9842；最终备份输出校验通过 | 独立读回GitHub保存输出，并非Drive权重独立下载验真；没有金融立场准确率 |
| 用户 T4 金融领域小实验 | 74步／1,211,960有效tokens；金融验证loss 3.2375、seed保留集3.9842→4.0574；最终备份输出校验通过 | 用户粘贴完整Colab输出；金融与通用验证集不同；不是25M预算完成或金融立场准确率 |
| 本地测试集 | 274项通过，557.43秒；本轮新增30项金融运行、2项Notebook、3项无回放和7项审核包测试 | Windows、Python3.12.14、CPU PyTorch2.14.1；包含真实持续运行Jupyter内核的安装、训练、备份、重复运行及新内核恢复 |
| Notebook 工程检查 | 14本格式、语法与无存储输出检查通过；01b/02c/02d/03a代码单元格实际执行 | 03a在两个隔离内核完成微型CPU母检查点读取、权重承接与优化器重置、1→3步严格续训、诊断、备份及新目录恢复。Git/Drive/外部下载使用本地替代；不是新金融阶段T4训练 |
| 扩充数据实际构建 | 544.91秒；采样RSS峰值344.7MiB；候选、清洗及token产物合计560.2MiB | 本地CPU、0.1秒采样；不含旧seed、检查点及外部缓存；不是精确分配峰值或Colab/T4显存测量 |
| 小模型完整链 | 自训tokenizer；预训练/domain/SFT/DPO各2次更新；DPO恢复到第3次；校准/评测/预测成功 | 126,912参数、441token词表、自编合成样本；不能评价金融能力 |
| 正式24M架构 | 24,255,744参数；CPU上一次前向、反向和AdamW更新成功 | batch1、长度16、随机token；不是正式语料预训练 |
| 恢复一致性 | CPU/dropout场景，模型、优化器、调度器、RNG和下一批与连续运行一致 | 精确恢复绑定版本、设备类型与数据配置；没有跨硬件逐位保证 |
| 断点/备份失败 | 损坏/不完整检查点被拒绝；模拟中途保存失败可恢复上一完整版本；训练中断前的检查点已定期备份 | 故障场景为本地目录模拟；用户另提供真实 Drive 最终备份成功输出，未独立核对云端文件 |
| 依赖 | `pip check`通过 | 本地安装清单见reports；Colab保留其CUDA PyTorch并另记实际版本 |

机器可读证据：`reports/cpu_smoke_check.json`、`reports/full_config_cpu_check.json`、`reports/environment-local-cpu.txt`、`reports/seed_continuation_checks.json` 记录工程检查；[扩充报告](../reports/expansion_local_validation_20261011.json) 记录本地实际下载、构建及资源测量；[02d续训工程检查](../reports/expanded_continuation_checks_20261011.json) 记录初次交付时232项测试、内核执行、数值一致性和恢复保护；`reports/colab_t4_continuation_20261011.json` 与 `reports/colab_t4_expansion_20261011.json` 记录用户提供的真实语料训练及备份输出。均没有金融立场测试分数。

2026-10-30本轮[金融衔接工程记录](../reports/domain_handoff_checks_20261030.json)补充274项完整测试、真实内核执行、带dropout数值一致性、AMP跳步无回放保护和真实本地混合数据来源核对。历史02d执行输出保留，格式失败修复已纳入本次回归。

## 已实现的功能

Dense/GQA/RoPE/QK-RMSNorm模型；可逆byte-level BPE；分组切分和词法近重复过滤；训练/评测隔离；uint16分片与mmap；精确80/20 token混合；全参数AdamW训练；FP16 AMP路径；标签位置SFT与DPO；完整恢复与定期检查点备份；目标输入规则；温度校准和独立阈值选择；JSONL预测；多数类和目标感知TF-IDF基线。

CLI 的流式入口及首批来源下载脚本已执行：Wikipedia 固定版本子集、美联储原创会议纪要、作者发布的 SEntFiN 候选池。来源条款、版本和哈希已记录；没有抓取 X 或下载整个大型语料库。

## 尚未完成

- 已完成有界金融领域阶段，长期预算所需的新语料和后续训练仍未完成。
- 完整预算所需的更广语料、目标级人工标注与资产映射、语义/事件级重复审查。
- 100M＋25M预算训练；首批8192词表已生成。后续语料扩充优先复用词表，需显式记录数据变更与检查点兼容策略；不能把不同数据伪装为精确恢复。
- 真实SFT/DPO、同新增标注的SFT对照、FinBERT比较、预训练消融、重复seed与事件组bootstrap。
- 对最终模型的独立校准、封存测试和实际使用验收。

01b／02c／02d／03a已完成。用户交付2000条CSV并明确确认其中1993条标题级标签，另7条待查；人工确认来源及完整数据身份校验见[审核记录](../reports/human_headline_review_20261030.json)。1879条三分类候选涉及740个资产候选组，资产身份仍未核验。下一步见[交接说明](POST_DOMAIN_NEXT.md)：核验资产、来源和事件分组，建立真实SFT分区。100M／25M完整预算及真实SFT/DPO、校准和封存测试均未完成。
