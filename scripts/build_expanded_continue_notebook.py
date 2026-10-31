"""Generate the strict same-data 610 to 1220 expansion continuation notebook."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 02d | Strictly resume expanded pretraining from 610 to 1,220 updates

        130U-FinLLM | 130U Financial Language Model

        Complete 02c, confirming 610 cumulative expansion updates, retention before/after
        and a final `verified: true` backup. This notebook uses the same expanded general-training data and original 8K vocabulary,
        restoring weights, optimizer, learning-rate schedule, RNG state and data cursor without reinitializing the model.
        There is no need to repeat 01b or 02c.

        By default verify cumulative update 610 to 611, then resume to 1,220. This adds 610 updates,
        nominally about 9,994,240 token positions, bringing the expansion stage to about 19,988,480 positions. Actual effective tokens
        are calculated from checkpoints and reports. Keep the original seed's approximately 4.91M tokens separate from this stage.
        This extends the stopping budget while retaining the original 6,104-update learning-rate schedule; it does not complete the 100M-token budget.

        Select Runtime > Change runtime type > T4 GPU and run cells in order. Do not launch a second training process concurrently.
        After disconnection, reconnect and check the original process status. To recover, rerun from the first cell,
        without deleting checkpoints, disabling identity checks or switching to random initialization to bypass errors.
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
            raise RuntimeError("Select a T4 GPU runtime; continued training does not start on CPU.")
        from finllm.expansion_continuation import (
            prepare_expansion_continuation, run_expansion_continuation,
            retention_continuation_check, backup_expansion_continuation)
        continuation = prepare_expansion_continuation(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION,
            completed_updates=COMPLETED_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        print("Restored cumulative expansion updates:", continuation["restored_updates"])
        print("Original 610-update artifacts preserved:", continuation["predecessor_archive"])
        print("Strict-resume configuration:", json.dumps(continuation["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown('''
        Read the frozen 610-update checkpoint diagnostics on original seed general-validation positions, then at 1,220 cumulative updates
        measure with the same tokenizer, data and positions. Retain expansion-validation loss separately. Both measure text prediction,
        not bullish/bearish accuracy. Reuse original reports when all input identities match.
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
            raise RuntimeError("Complete original-validation diagnostics for the 610-update checkpoint first.")
        continuation_probe = run_expansion_continuation(continuation, "probe", finllm)
        print(json.dumps(continuation_probe, indent=2, ensure_ascii=False))
        CONTINUATION_PROBE_SUCCEEDED = True
        '''), markdown('''
        Continue to 1,220 expansion updates only after one-step recovery succeeds. For later same-version checkpoints within the target range, resume from the actual cursor
        rather than rolling back to update 610. Save and sync every 10 updates; retain original 610-update artifacts, checkpoint identity and
        retention reports in a separate archive. Record elapsed time and throughput per call; rerunning a completed stage preserves original measurements.
        Incompatible data/tokenizer/configuration/runtime stops execution. The stopping budget may increase while schedule semantics remain unchanged.
        '''), code('''
        CONTINUATION_TRAIN_SUCCEEDED = False
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_RETENTION_BEFORE_SUCCEEDED", False) or not globals().get("CONTINUATION_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete original-validation diagnostics and one-step strict-recovery verification first.")
        continuation_result = run_expansion_continuation(continuation, "continue", finllm)
        print(json.dumps(continuation_result, indent=2, ensure_ascii=False))
        CONTINUATION_TRAIN_SUCCEEDED = True
        '''), code('''
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_TRAIN_SUCCEEDED", False):
            raise RuntimeError("This continuation segment has not completed successfully; keep existing backups.")
        continuation_retention_after = retention_continuation_check(continuation, "after")
        print(json.dumps(continuation_retention_after, indent=2, ensure_ascii=False))
        CONTINUATION_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        CONTINUATION_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "CONTINUATION_RETENTION_BEFORE_SUCCEEDED", "CONTINUATION_PROBE_SUCCEEDED",
            "CONTINUATION_TRAIN_SUCCEEDED", "CONTINUATION_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("Diagnostics, one-step verification and continued training have not all succeeded; do not report a complete stage backup.")
        continuation_backup = backup_expansion_continuation(continuation)
        print(json.dumps(continuation_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", continuation["continuation_drive"] / "runtime_revision.txt")
        CONTINUATION_BACKED_UP = True
        print("Expanded general continuation and verified backup complete:", continuation["paths"]["drive"] / "run")
        print("Segment reports and diagnostics directory:", continuation["continuation_drive"])
        print("Original 610-update artifact archive:", continuation["predecessor_archive"])
        print("Original seed data and 300-update artifacts preserved:", continuation["paths"]["seed_drive"])
        '''), markdown('''
        Completion criteria: new one-step and training-segment reports, 1,220 cumulative updates, actual new/cumulative tokens,
        retention before/after and a final `verified: true` backup. Keep these outputs to inform the next stage.
        After entering this segment, use 02d to resume. Preserve the original 02c 610-update report and measurements in a separate archive.
        This notebook continues general pretraining without starting financial-domain training, SFT or DPO; it does not evaluate asset stance or scoring ability.
        ''')]
    write("02d_expanded_continue.ipynb", cells)


if __name__ == "__main__":
    build()
