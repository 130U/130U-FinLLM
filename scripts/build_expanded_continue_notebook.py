"""Generate the strict same-data 610 to 1220 expansion continuation notebook."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 02d｜从扩充预训练 610 步严格续训到 1220 步

        130U-FinLLM｜130U 金融大语言模型

        先完成 02c，确认训练达到扩充阶段累计 610 步，并得到 retention before/after
        与最终 `verified: true` 备份。本页使用相同的扩充通用训练数据与原 8K 词表，
        恢复现有权重、优化器、学习率调度、随机数状态和数据游标；不会重新初始化模型。
        不需要重做 01b 或 02c。

        默认先验证累计 610→611 步，再续训到累计 1220 步。本段共新增 610 步，
        名义约 9,994,240 token 位置；扩充阶段累计约 19,988,480 个位置，实际有效 tokens
        由检查点和报告计算。原 seed 的约 4.91M tokens 单独保留，不混入本阶段计数。
        本页延长停止预算，保持原 6104 步学习率调度跨度，不表示 100M 预算已完成。

        选择「运行时 → 更改运行时类型 → T4 GPU」，逐格执行。不要并行启动第二个训练进程。
        连接断开时先等重新连接、确认原进程状态；需要恢复时从第一格重新执行，
        不要删除检查点、取消身份校验或换成随机初始化来绕过报错。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        CONTINUATION_RETENTION_BEFORE_SUCCEEDED = False
        CONTINUATION_PROBE_SUCCEEDED = False
        CONTINUATION_TRAIN_SUCCEEDED = False
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        COMPLETED_UPDATES = 610
        PROBE_UPDATES = 611
        TARGET_UPDATES = 1220
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 运行时；正式续训不在 CPU 上启动。")
        from finllm.expansion_continuation import (
            prepare_expansion_continuation, run_expansion_continuation,
            retention_continuation_check, backup_expansion_continuation)
        continuation = prepare_expansion_continuation(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION,
            completed_updates=COMPLETED_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        print("已恢复扩充阶段累计步数：", continuation["restored_updates"])
        print("已保留 610 步原始成果：", continuation["predecessor_archive"])
        print("本次严格续训配置：", json.dumps(continuation["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown('''
        先读取已冻结 610 步检查点在原 seed 通用验证位置上的训练前诊断，再于累计 1220 步
        使用相同 tokenizer、数据及位置测量。另行保留扩充验证集 loss。两者均衡量文本预测，
        不能当作看多／看空准确率。已有输入身份完全一致的测量时复用原始报告。
        '''), code('''
        CONTINUATION_RETENTION_BEFORE_SUCCEEDED = False
        CONTINUATION_PROBE_SUCCEEDED = False
        CONTINUATION_TRAIN_SUCCEEDED = False
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        continuation_retention_before = retention_continuation_check(continuation, "before")
        print(json.dumps(continuation_retention_before, indent=2, ensure_ascii=False))
        CONTINUATION_RETENTION_BEFORE_SUCCEEDED = True
        '''), code('''
        CONTINUATION_PROBE_SUCCEEDED = False
        CONTINUATION_TRAIN_SUCCEEDED = False
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_RETENTION_BEFORE_SUCCEEDED", False):
            raise RuntimeError("请先成功完成 610 步检查点的原验证集诊断。")
        continuation_probe = run_expansion_continuation(continuation, "probe", finllm)
        print(json.dumps(continuation_probe, indent=2, ensure_ascii=False))
        CONTINUATION_PROBE_SUCCEEDED = True
        '''), markdown('''
        单步恢复成功后才运行到扩充阶段累计 1220 步。已有目标范围内更靠后的同版本检查点时按实际游标
        恢复，不回退到 610 步。每 10 步保存并同步 Drive；原 610 步成果、检查点身份和
        retention 报告保存在独立归档。各次调用耗时与吞吐单独记录，重复完成阶段保留原测量。
        数据／tokenizer／配置／运行时不兼容会明确停止；停止预算可以增加，调度语义保持不变。
        '''), code('''
        CONTINUATION_TRAIN_SUCCEEDED = False
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_RETENTION_BEFORE_SUCCEEDED", False) or not globals().get("CONTINUATION_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先完成原验证集诊断和单步严格恢复验证。")
        continuation_result = run_expansion_continuation(continuation, "continue", finllm)
        print(json.dumps(continuation_result, indent=2, ensure_ascii=False))
        CONTINUATION_TRAIN_SUCCEEDED = True
        '''), code('''
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_TRAIN_SUCCEEDED", False):
            raise RuntimeError("本段续训尚未成功结束；保留已有备份。")
        continuation_retention_after = retention_continuation_check(continuation, "after")
        print(json.dumps(continuation_retention_after, indent=2, ensure_ascii=False))
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        CONTINUATION_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "CONTINUATION_RETENTION_BEFORE_SUCCEEDED", "CONTINUATION_PROBE_SUCCEEDED",
            "CONTINUATION_TRAIN_SUCCEEDED", "CONTINUATION_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("诊断／单步验证／续训尚未全部成功；不得显示完整阶段备份。")
        continuation_backup = backup_expansion_continuation(continuation)
        print(json.dumps(continuation_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", continuation["continuation_drive"] / "runtime_revision.txt")
        CONTINUATION_BACKED_UP = True
        print("扩充通用续训与校验备份完成：", continuation["paths"]["drive"] / "run")
        print("本段报告与诊断目录：", continuation["continuation_drive"])
        print("610 步原始成果归档：", continuation["predecessor_archive"])
        print("原 seed 数据与 300 步成果保留：", continuation["paths"]["seed_drive"])
        '''), markdown('''
        完成标准：新单步与正式段报告、累计 1220 步、实际新增／累计 tokens、
        retention before/after，以及 `verified: true` 最终备份。保留这些输出用于下一阶段判断。
        进入本段后使用 02d 恢复进度；原 02c 的 610 步报告及测量作为独立归档保留。
        本页继续通用预训练，未启动金融领域训练、SFT 或 DPO；资产立场与评分能力尚未评估。
        ''')]
    write("02d_expanded_continue.ipynb", cells)


if __name__ == "__main__":
    build()
