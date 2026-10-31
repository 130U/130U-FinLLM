"""Generate a Colab frontend for the reviewed literal-entity SFT pilot."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04a | Entity-stance SFT pilot from the financial-domain model

        Use the 74-update `domain_v1` checkpoint and original 8K tokenizer from Drive; the first handoff imports weights and creates a new optimizer.
        Train on human-confirmed headline stance toward literal entities in the text. This is a private research pilot; targets are not verified security identifiers.
        Semantic-event and historical-time audits are incomplete; development metrics do not measure real Twitter or future-market accuracy.
        Source articles may carry third-party rights. Keep this material within personal research and preserve source notices.

        Select T4 and run cells in order. Upload local `entity_v1_upload.zip` initially; subsequent runs reuse the Drive backup.
        After disconnection, confirm the previous process has stopped before rerunning from the beginning. Record the 40-character commit shown in the first cell; if main changes,
        set REF to that original full commit before resuming. Do not start two training processes concurrently.
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.sft_pilot_runtime import install_task_zip, prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
        from finllm.expansion_runtime import _inventory
        TASK_VERSION = "entity_v1"
        RUN_VERSION = "sft_entity_v1"
        TASK_ZIP_SHA256 = "e87d12013271eb8f93f926b0b2c71cef15b0c777cc58d6e08348773dbd6a7be6"
        TASK_DRIVE = DRIVE_ROOT / "tasks" / TASK_VERSION
        if not (TASK_DRIVE / "SNAPSHOT_COMPLETE.json").exists():
            from google.colab import files
            import tempfile
            with tempfile.TemporaryDirectory(prefix="finllm-upload-") as folder:
                original_cwd = Path.cwd()
                try:
                    os.chdir(folder)
                    print("Select local entity_v1_upload.zip; it will be verified and backed up to Drive.")
                    uploaded = files.upload()
                    if len(uploaded) != 1:
                        raise ValueError("Select exactly one task ZIP.")
                    name = next(iter(uploaded))
                    if Path(name).name != name:
                        raise ValueError("Invalid upload filename.")
                    result = install_task_zip(Path(folder) / name, DRIVE_ROOT, TASK_ZIP_SHA256, TASK_VERSION)
                finally:
                    os.chdir(original_cwd)
            print("Task-data backup:", json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _inventory(TASK_DRIVE)
            print("Task-data snapshot reused and verified:", TASK_DRIVE)
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("Select T4 GPU, then rerun from the first cell.")
        pilot = prepare_sft_pilot(PROJECT, DRIVE_ROOT, task_version=TASK_VERSION, run_version=RUN_VERSION)
        print("Verified domain parent model:", json.dumps(pilot["parent_audit"], indent=2, ensure_ascii=False))
        print("Task partition counts:", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        print("Current GPU:", torch.cuda.get_device_name(0))
        '''), code('''
        SFT_PROBE_SUCCEEDED = False
        SFT_TRAIN_SUCCEEDED = False
        SFT_BACKED_UP = False
        probe = run_sft_pilot(pilot, "probe", finllm)
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        SFT_PROBE_SUCCEEDED = True
        '''), code('''
        SFT_TRAIN_SUCCEEDED = False
        SFT_BACKED_UP = False
        if not globals().get("SFT_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete the one-step SFT verification first.")
        result = run_sft_pilot(pilot, "continue", finllm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        SFT_TRAIN_SUCCEEDED = True
        '''), code('''
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False):
            raise RuntimeError("Complete SFT training first.")
        SFT_OUTPUT = Path(pilot["config"]["output_dir"])
        SFT_DEV = pilot["task_local"] / "development.jsonl"
        for name, checkpoint in (("domain_before", Path(pilot["config"]["init_checkpoint"])),
                                 ("sft_best", SFT_OUTPUT / "best.pt")):
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", pilot["config"]["tokenizer"],
                   "--input", SFT_DEV, "--output", SFT_OUTPUT / (name + "_development.json"),
                   "--target-mode", "literal_entity")
        SFT_EVAL_SUCCEEDED = True
        print("Saved evaluations of the domain model and best SFT checkpoint on the same development set.")
        '''), code('''
        SFT_BACKED_UP = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False) or not globals().get("SFT_EVAL_SUCCEEDED", False):
            raise RuntimeError("SFT training or the development comparison is incomplete; preserve existing checkpoints.")
        backup = backup_sft_pilot(pilot)
        print(json.dumps(backup, indent=2, ensure_ascii=False))
        SFT_BACKED_UP = True
        print("Entity pilot complete and backed up:", pilot["stage_drive"] / "run")
        '''), markdown('''
        Keep training reports, development macro-F1 and `verified: true` backup output.
        Small-sample results may vary. This notebook does not complete final testing, asset-identifier mapping, DPO preference review, calibration or real X evaluation.
        ''')]
    write("04a_entity_sft_pilot.ipynb", cells)


if __name__ == "__main__":
    build()
