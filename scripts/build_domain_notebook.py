"""Generate a versioned financial-domain handoff from completed pretraining."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 03a｜承接已完成预训练的金融领域训练

        130U-FinLLM｜130U 金融大语言模型

        本页先读取并校验 Drive 中至少完成扩充阶段 1220 步的检查点、数据和原 8K 词表，
        将母检查点冻结后才允许训练。默认复用 `expansion_v1/domain_mix`，
        其 1,215,545 tokens 包含约 80% 金融／20% 通用内容。

        这是新的 `domain_v1` 阶段：首次只承接母模型权重，建立新的优化器和学习率调度，
        阶段更新从 0 开始。已有本阶段检查点时严格恢复完整状态，不重新初始化。
        选择「运行时 → 更改运行时类型 → T4 GPU」，从第一格顺序运行。

        默认先验证第 1 步，再训练到本阶段累计 74 步，名义约 1,212,416 个 token 位置。
        这是现有语料的一次有界小实验；不是 25M 领域预算已经完成，也不代表资产立场准确率。
        不需要重做 01b、02c 或 02d，不要同时启动另一个训练进程。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        DOMAIN_INPUTS_VERIFIED = False
        DOMAIN_RETENTION_BEFORE_SUCCEEDED = False
        DOMAIN_PROBE_SUCCEEDED = False
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        DOMAIN_VERSION = "domain_v1"
        MIN_PARENT_UPDATES = 1220
        PROBE_UPDATES = 1
        TARGET_UPDATES = 74
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 运行时；正式领域训练需要 GPU。")
        from finllm.domain_runtime import (
            prepare_domain_training, run_domain_segment,
            retention_domain_check, backup_domain_training)
        from finllm.checkpoint import load_checkpoint, sha256_file
        domain = prepare_domain_training(
            PROJECT, DRIVE_ROOT, DOMAIN_VERSION, EXPANSION_VERSION, SEED_VERSION,
            min_parent_updates=MIN_PARENT_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        # 直接读已校验并冻结的母检查点，不把旧截图当成当前 Drive 读取结果。
        parent_checkpoint = Path(domain["config"]["init_checkpoint"])
        parent_payload = load_checkpoint(parent_checkpoint, allow_previous=False)
        parent_audit = {
            "frozen_checkpoint": str(parent_checkpoint),
            "checkpoint_sha256": sha256_file(parent_checkpoint),
            "stage": parent_payload["config"]["stage"],
            "updates": parent_payload["progress"]["updates"],
            "tokens": parent_payload["progress"]["tokens"],
            "tokenizer_sha256": parent_payload["tokenizer_hash"],
            "data_sha256": parent_payload["provenance"]["data_sha256"],
            "validation_sha256": parent_payload["provenance"]["validation_sha256"],
        }
        if parent_audit["stage"] != "pretrain" or parent_audit["updates"] < MIN_PARENT_UPDATES:
            raise ValueError("母检查点不是要求的已完成预训练成果；保留已有文件并停止。")
        print("当前已校验的 Drive 母检查点：", json.dumps(parent_audit, indent=2, ensure_ascii=False))
        print("当前领域阶段累计步数：", domain["restored_updates"])
        print("本阶段配置：", json.dumps(domain["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        DOMAIN_INPUTS_VERIFIED = True
        '''), markdown('''
        上一格只检查输入、冻结母成果并准备配置，尚未启动优化器更新。
        保留其实际母检查点 SHA256、步数和 tokens，作为本次云端读取证据。
        下方诊断使用冻结的验证文本；诊断和训练 loss 衡量下一 token 预测，
        不应解释为看多／看空分数。验证失败时后续训练和完整备份会停止。
        '''), code('''
        DOMAIN_RETENTION_BEFORE_SUCCEEDED = False
        DOMAIN_PROBE_SUCCEEDED = False
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_INPUTS_VERIFIED", False):
            raise RuntimeError("请先完成当前 Drive 输入校验和母检查点冻结。")
        domain_retention_before = retention_domain_check(domain, "before")
        print(json.dumps(domain_retention_before, indent=2, ensure_ascii=False))
        DOMAIN_RETENTION_BEFORE_SUCCEEDED = True
        '''), code('''
        DOMAIN_PROBE_SUCCEEDED = False
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_INPUTS_VERIFIED", False) or not globals().get("DOMAIN_RETENTION_BEFORE_SUCCEEDED", False):
            raise RuntimeError("请先完成母检查点校验和训练前诊断。")
        domain_probe = run_domain_segment(domain, "probe", finllm)
        print(json.dumps(domain_probe, indent=2, ensure_ascii=False))
        DOMAIN_PROBE_SUCCEEDED = True
        '''), markdown('''
        单步成功后继续到本阶段累计 74 步。母预训练与领域阶段的 tokens 分别记录，
        沿模型路径的累计量另列。原 seed、扩充训练、1220 步母检查点及其报告保持原样。
        首次领域权重承接与后续严格恢复分别处理；数据、词表、配置或本阶段运行时不兼容会停止。
        连接断开后先确认旧执行状态，再从本页第一格恢复，保留已有检查点和报告。
        '''), code('''
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先成功完成领域阶段单步验证。")
        domain_result = run_domain_segment(domain, "continue", finllm)
        print(json.dumps(domain_result, indent=2, ensure_ascii=False))
        DOMAIN_TRAIN_SUCCEEDED = True
        '''), code('''
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_TRAIN_SUCCEEDED", False):
            raise RuntimeError("本阶段训练尚未成功结束；保留已有备份。")
        domain_retention_after = retention_domain_check(domain, "after")
        print(json.dumps(domain_retention_after, indent=2, ensure_ascii=False))
        DOMAIN_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        DOMAIN_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "DOMAIN_INPUTS_VERIFIED", "DOMAIN_RETENTION_BEFORE_SUCCEEDED",
            "DOMAIN_PROBE_SUCCEEDED", "DOMAIN_TRAIN_SUCCEEDED", "DOMAIN_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("输入／诊断／训练尚未全部成功；不得显示完整阶段备份。")
        domain_backup = backup_domain_training(domain)
        print(json.dumps(domain_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", domain["paths"]["drive"] / "runtime_revision.txt")
        DOMAIN_BACKED_UP = True
        print("金融领域训练与校验备份完成：", domain["paths"]["drive"] / "run")
        print("领域阶段报告与诊断：", domain["paths"]["drive"])
        print("完整母检查点已冻结：", parent_checkpoint)
        '''), markdown('''
        完成标准：实际母检查点身份已读取；领域单步与累计 74 步报告可核验；
        训练前后诊断保留；最后备份为 `verified: true`。
        已完成阶段重复运行会复用原检查点与真实测量，不重新计时或覆盖原报告。
        本页完成有限金融语料实验，原 100M／25M 计算预算仍单独记录。
        正式 SFT、DPO、校准和资产立场评测继续需要符合本项目定义的目标级标签与审核记录。
        ''')]
    write("03a_domain_pretrain.ipynb", cells)


if __name__ == "__main__":
    build()
