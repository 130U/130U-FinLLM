# 实现与验证状态

更新日期：2026-10-09。

## 已执行的验证

| 检查 | 实际结果 | 边界 |
|---|---|---|
| 本地测试集 | 52项通过 | Windows、Python3.12.14、CPU PyTorch2.14.1；不是T4验证 |
| 7本Notebook | nbformat及所有代码单元格Python语法检查通过，输出为空 | 尚未在用户Colab执行 |
| 小模型完整链 | 自训tokenizer；预训练/domain/SFT/DPO各2次更新；DPO恢复到第3次；校准/评测/预测成功 | 126,912参数、441token词表、自编合成样本；不能评价金融能力 |
| 正式24M架构 | 24,255,744参数；CPU上一次前向、反向和AdamW更新成功 | batch1、长度16、随机token；不是正式语料预训练 |
| 恢复一致性 | CPU/dropout场景，模型、优化器、调度器、RNG和下一批与连续运行一致 | 精确恢复绑定版本、设备类型与数据配置；没有跨硬件逐位保证 |
| 断点/备份失败 | 损坏/不完整检查点被拒绝；模拟中途保存失败可恢复上一完整版本；训练中断前的检查点已定期备份 | 本地目录模拟持久盘；Drive实际挂载仍需用户运行 |
| 依赖 | `pip check`通过 | 本地安装清单见reports；Colab保留其CUDA PyTorch并另记实际版本 |

机器可读证据：`reports/cpu_smoke_check.json`、`reports/full_config_cpu_check.json`、`reports/environment-local-cpu.txt`。这些文件只记录工程验证，没有真实金融测试分数。

## 已实现的功能

Dense/GQA/RoPE/QK-RMSNorm模型；可逆byte-level BPE；分组切分和词法近重复过滤；训练/评测隔离；uint16分片与mmap；精确80/20 token混合；全参数AdamW训练；FP16 AMP路径；标签位置SFT与DPO；完整恢复与定期检查点备份；目标输入规则；温度校准和独立阈值选择；JSONL预测；多数类和目标感知TF-IDF基线。

CLI的流式数据下载入口已实现，尚未替用户下载正式大型语料或取得第三方数据授权。

## 尚未完成

- 用户Colab上T4实跑、峰值allocated/reserved显存及稳定吞吐测量。
- 正式数据取得、权限确认、目标级人工标注、语义/事件级重复审查。
- 正式8192词表训练及100M＋25M预算训练。
- 真实SFT/DPO、同新增标注的SFT对照、FinBERT比较、预训练消融、重复seed与事件组bootstrap。
- 对最终模型的独立校准、封存测试和实际使用验收。

README第一本Notebook是下一步入口。正式阶段默认不开跑，需要真实数据、已挂载Drive与明确训练配置。不要把测试通过、合成样本loss下降或一个随机token更新写成“金融模型已训练完成”。
