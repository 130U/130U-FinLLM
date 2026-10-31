"""Generate a versioned financial-domain handoff from completed pretraining."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 03a | Financial-domain training from completed pretraining

        130U-FinLLM | 130U Financial Language Model

        Read and verify the Drive checkpoint with at least 1,220 expansion updates, its data and the original 8K tokenizer,
        and freeze the parent checkpoint before training. By default reuse `expansion_v1/domain_mix`,
        whose 1,215,545 tokens contain approximately 80% financial and 20% general content.

        This is a new `domain_v1` stage: initially import only parent weights and create a new optimizer and learning-rate schedule,
        starting stage updates at zero. Existing checkpoints for this stage restore complete state strictly, without reinitialization.
        Select Runtime > Change runtime type > T4 GPU and run in order from the first cell.

        By default verify update 1, then train to 74 cumulative stage updates, nominally about 1,212,416 token positions.
        This is a bounded experiment on existing corpora; it neither completes the 25M domain budget nor measures asset-stance accuracy.
        Do not repeat 01b, 02c or 02d, or start another training process concurrently.
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
            raise RuntimeError("Select a T4 GPU runtime; domain training requires a GPU.")
        from finllm.domain_runtime import (
            prepare_domain_training, run_domain_segment,
            retention_domain_check, backup_domain_training)
        from finllm.checkpoint import load_checkpoint, sha256_file
        domain = prepare_domain_training(
            PROJECT, DRIVE_ROOT, DOMAIN_VERSION, EXPANSION_VERSION, SEED_VERSION,
            min_parent_updates=MIN_PARENT_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        # Read the verified frozen parent checkpoint directly; an old screenshot is not a current Drive read.
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
            raise ValueError("The parent checkpoint is not the required completed pretraining artifact; preserve existing files and stop.")
        print("Currently verified Drive parent checkpoint:", json.dumps(parent_audit, indent=2, ensure_ascii=False))
        print("Current cumulative domain updates:", domain["restored_updates"])
        print("Stage configuration:", json.dumps(domain["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        DOMAIN_INPUTS_VERIFIED = True
        '''), markdown('''
        The previous cell only validates inputs, freezes parent artifacts and prepares configuration; optimizer updates have not started.
        Keep the actual parent SHA256, update count and tokens as evidence of this cloud read.
        The diagnostics below use frozen validation texts. Diagnostic and training loss measure next-token prediction,
        not bullish/bearish scores. Validation failure prevents subsequent training and complete backups.
        '''), code('''
        DOMAIN_RETENTION_BEFORE_SUCCEEDED = False
        DOMAIN_PROBE_SUCCEEDED = False
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_INPUTS_VERIFIED", False):
            raise RuntimeError("Validate current Drive inputs and freeze the parent checkpoint first.")
        domain_retention_before = retention_domain_check(domain, "before")
        print(json.dumps(domain_retention_before, indent=2, ensure_ascii=False))
        DOMAIN_RETENTION_BEFORE_SUCCEEDED = True
        '''), code('''
        DOMAIN_PROBE_SUCCEEDED = False
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_INPUTS_VERIFIED", False) or not globals().get("DOMAIN_RETENTION_BEFORE_SUCCEEDED", False):
            raise RuntimeError("Validate the parent checkpoint and complete pretraining diagnostics first.")
        domain_probe = run_domain_segment(domain, "probe", finllm)
        print(json.dumps(domain_probe, indent=2, ensure_ascii=False))
        DOMAIN_PROBE_SUCCEEDED = True
        '''), markdown('''
        After one-step verification, continue to 74 cumulative stage updates. Record parent pretraining and domain tokens separately,
        with a separate total along the model lineage. Preserve original seed and expansion artifacts, the 1,220-update parent checkpoint and its reports.
        Handle initial domain weight transfer separately from strict recovery. Incompatible data, tokenizer, configuration or stage runtime stops execution.
        After disconnection, check prior execution status before recovering from the first cell; preserve existing checkpoints and reports.
        '''), code('''
        DOMAIN_TRAIN_SUCCEEDED = False
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete the domain-stage one-step check successfully first.")
        domain_result = run_domain_segment(domain, "continue", finllm)
        print(json.dumps(domain_result, indent=2, ensure_ascii=False))
        DOMAIN_TRAIN_SUCCEEDED = True
        '''), code('''
        DOMAIN_RETENTION_AFTER_SUCCEEDED = False
        DOMAIN_BACKED_UP = False
        if not globals().get("DOMAIN_TRAIN_SUCCEEDED", False):
            raise RuntimeError("Stage training has not completed successfully; keep existing backups.")
        domain_retention_after = retention_domain_check(domain, "after")
        print(json.dumps(domain_retention_after, indent=2, ensure_ascii=False))
        DOMAIN_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        DOMAIN_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "DOMAIN_INPUTS_VERIFIED", "DOMAIN_RETENTION_BEFORE_SUCCEEDED",
            "DOMAIN_PROBE_SUCCEEDED", "DOMAIN_TRAIN_SUCCEEDED", "DOMAIN_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("Input checks, diagnostics and training have not all succeeded; do not report a complete stage backup.")
        domain_backup = backup_domain_training(domain)
        print(json.dumps(domain_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", domain["paths"]["drive"] / "runtime_revision.txt")
        DOMAIN_BACKED_UP = True
        print("Financial-domain training and verified backup complete:", domain["paths"]["drive"] / "run")
        print("Domain-stage reports and diagnostics:", domain["paths"]["drive"])
        print("Complete parent checkpoint frozen:", parent_checkpoint)
        '''), markdown('''
        Completion criteria: actual parent identity has been read; domain one-step and cumulative 74-update reports are verifiable;
        before/after diagnostics are retained; the final backup reports `verified: true`.
        Rerunning a completed stage reuses original checkpoints and measurements without timing it again or overwriting reports.
        This notebook completes a bounded financial-corpus experiment; the original 100M/25M compute budgets remain separate.
        SFT, DPO, calibration and asset-stance evaluation require target-level labels and review records consistent with the project task definition.
        ''')]
    write("03a_domain_pretrain.ipynb", cells)


if __name__ == "__main__":
    build()
