"""Generate the data expansion and explicit weights-transfer Colab notebooks."""
from build_notebooks import BOOTSTRAP, markdown, code, write


DATA_BOOTSTRAP = BOOTSTRAP.replace('"-q", "-e", ".",', '"-q", "-e", ".[data]",')
if DATA_BOOTSTRAP == BOOTSTRAP:
    raise RuntimeError("Shared bootstrap install transaction could not be located")

DRIVE = '''
MOUNT_DRIVE = True
if not MOUNT_DRIVE:
    raise RuntimeError("This stage requires existing seed inputs and versioned backups; mount Drive first.")
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print("Drive project directory:", DRIVE_ROOT)
'''


def build():
    cells = [markdown('''
        # 01b | Expand English general and financial corpora with the original 8K tokenizer

        130U-FinLLM | 130U Financial Language Model

        Preserve completed seed_v1 data, tokenizer and 300-update artifacts. This notebook creates a separate expansion_v1:
        use up to 200 MB by default from a pinned Wikipedia snapshot and extend FOMC minutes to 2011-2025; exclude original seed documents and frozen task texts,
        split new train/dev data by document and encode with the original tokenizer. Report actual independent document and token counts after completion.
        No bullish/bearish labels, tokenizer retraining or model training are performed. Source texts stay in this runtime and your own Drive.

        CPU is sufficient. Run cells in order; after disconnection, reconnect and confirm execution has stopped before rerunning.
        Reruns validate and reuse complete artifacts for the same version; if parameters or inputs change, preserve the old version and choose a new version name.
        Downloads and Parquet decoding use additional caches; the 200 MB output cap is not a limit on total memory or disk use.
        See docs/research/expanded_corpus_sources_20261011.md for sources and rights.
        '''), code(DATA_BOOTSTRAP), code(DRIVE), code('''
        EXPANSION_FETCH_SUCCEEDED = False
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        GENERAL_MAX_BYTES = 200_000_000
        MAX_GENERAL_DOCUMENTS = 50_000
        FINANCE_YEARS = ",".join(str(year) for year in range(2011, 2026))
        FINANCE_MAX_BYTES = 12_000_000
        MAX_FINANCE_DOCUMENTS = 128
        from finllm.expansion_runtime import restore_expansion_materials, expansion_build_arguments, backup_expansion_data
        from finllm.expansion import build_expansion_artifacts
        expansion_paths = restore_expansion_materials(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION)
        print(json.dumps({key: str(value) for key, value in expansion_paths.items()}, indent=2, ensure_ascii=False))
        print("Data preparation only; CUDA available:", torch.cuda.is_available())
        '''), markdown('''
        Fetch bounded candidates. Validate completed source outputs read-only; after an interrupted download, validate progress and page caches before continuing.
        Leave seed_v1 unchanged. Stop at the byte/document limit; these limits do not guarantee 100M general or 25M independent financial tokens.
        '''), code('''
        EXPANSION_FETCH_SUCCEEDED = False
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        command(sys.executable, "-u", "scripts/fetch_expansion_corpora.py",
                "--output", expansion_paths["corpora"],
                "--general-max-bytes", GENERAL_MAX_BYTES,
                "--max-general-documents", MAX_GENERAL_DOCUMENTS,
                "--finance-years", FINANCE_YEARS,
                "--finance-max-bytes", FINANCE_MAX_BYTES,
                "--max-finance-documents", MAX_FINANCE_DOCUMENTS, cwd=PROJECT)
        print((expansion_paths["corpora"] / "manifest.json").read_text(encoding="utf-8"))
        EXPANSION_FETCH_SUCCEEDED = True
        '''), code('''
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_FETCH_SUCCEEDED", False):
            raise RuntimeError("Source acquisition has not succeeded; resolve the error above and preserve existing inputs.")
        from finllm.runtime import sync_snapshot
        source_backup = sync_snapshot(expansion_paths["corpora"], expansion_paths["drive"] / "corpora")
        print(json.dumps(source_backup, indent=2, ensure_ascii=False))
        EXPANSION_CORPORA_BACKED_UP = True
        print("Candidate sources verified and backed up; cleaning and encoding follow.")
        '''), markdown('''
        Use the original 8K tokenizer; exclude seed train/validation source identities, exact and lexical near-duplicates, and the frozen task pool.
        The new validation set is for language-model diagnostics; retain original seed validation shards for before/after loss comparisons in 02c.
        These remain unlabeled language-model corpora and do not measure asset-stance accuracy.
        '''), code('''
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_CORPORA_BACKED_UP", False):
            raise RuntimeError("Back up candidate sources successfully before building expansion artifacts.")
        expansion_data_result = build_expansion_artifacts(**expansion_build_arguments(expansion_paths))
        print(json.dumps(expansion_data_result, indent=2, ensure_ascii=False))
        print((expansion_paths["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
        EXPANSION_DATA_SUCCEEDED = True
        '''), code('''
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_DATA_SUCCEEDED", False):
            raise RuntimeError("Data construction has not succeeded; do not report partial artifacts as a complete backup.")
        expansion_backup = backup_expansion_data(expansion_paths)
        print(json.dumps(expansion_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", expansion_paths["drive"] / "runtime_revision.txt")
        EXPANSION_DATA_BACKED_UP = True
        print("Expanded data and verified backup complete:", expansion_paths["drive"])
        '''), markdown('''
        Completion criteria: expansion_summary.json records actual new documents/tokens; corpora, prepared and processed backups all report `verified: true`.
        Switch to T4 for 02c and explicitly initialize from completed seed 300-update weights. Do not repeat 01a/02a/02b or run the old 03 directly.
        If candidates cannot support 610 updates without replay, 02c stops and requests more corpus data; repeated text is not new independent data.
        ''')]
    write("01b_expand_data.ipynb", cells)

    cells = [markdown('''
        # 02c | Initialize from seed weights for a new expanded-corpus stage

        130U-FinLLM | 130U Financial Language Model

        Complete 01b and verify all three backups. This notebook requires a real seed_v1 pretraining checkpoint with at least 300 updates,
        freezes its identity and retains the original 8K tokenizer. Because the corpus changes, initialize from weights while creating new optimizer and scheduler states,
        data cursors and stage counts. Preserve the original seed checkpoint; this is not an exact resume of the same data stream.

        By default verify update 0 to 1, then train to 610 cumulative updates in the new stage, nominally about 10M new token positions.
        Reports provide actual effective tokens; the parent model's approximately 4.91M tokens are listed separately. The learning-rate schedule spans 6,104 updates,
        so reaching 610 updates does not complete the 100M-token budget or the full financial model.

        Select Runtime > Change runtime type > T4 GPU, then run cells in order.
        Rerun the same version with strict recovery. After interruption, reconnect and confirm training has stopped before rerunning from the first cell.
        Do not start a second training process concurrently. Use a separate experiment version when changing data, tokenizer or execution semantics.
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = False
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        MIN_PARENT_UPDATES = 300
        PROBE_UPDATES = 1
        TARGET_UPDATES = 610
        if not torch.cuda.is_available():
            raise RuntimeError("Select a T4 GPU runtime; the expansion stage does not start on CPU.")
        from finllm.expansion_runtime import prepare_expansion_training, run_expansion_segment, retention_check, backup_expansion_training
        expansion = prepare_expansion_training(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION,
            min_parent_updates=MIN_PARENT_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        print("Restored new-stage update count:", expansion["restored_updates"])
        print("Frozen parent checkpoint identity:", json.dumps(expansion["parent"], indent=2, ensure_ascii=False))
        print("New-stage configuration:", json.dumps(expansion["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown('''
        Measure bounded loss for the frozen parent checkpoint on original seed general-validation shards, then compare identical positions after training.
        Also retain validation loss for the expanded corpus. Both diagnose text prediction, not financial stance.
        Read existing measurements when all input identities match.
        '''), code('''
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = False
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        retention_before = retention_check(expansion, "before")
        print(json.dumps(retention_before, indent=2, ensure_ascii=False))
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = True
        '''), code('''
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_RETENTION_BEFORE_SUCCEEDED", False):
            raise RuntimeError("Complete the parent checkpoint measurement on the original validation set first.")
        expansion_probe = run_expansion_segment(expansion, "probe", finllm)
        print(json.dumps(expansion_probe, indent=2, ensure_ascii=False))
        EXPANSION_PROBE_SUCCEEDED = True
        '''), markdown('''
        Train to 610 new-stage updates only after the 0-to-1 check succeeds. Validate and restore later checkpoints of the same version without rolling back.
        Save and sync to Drive every 10 updates. Reported updates/tokens are cumulative within the new stage; elapsed time covers only that call.
        Incompatible data/tokenizer/configuration/runtime stops execution; do not disable provenance checks or use random initialization to bypass errors.
        '''), code('''
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_RETENTION_BEFORE_SUCCEEDED", False) or not globals().get("EXPANSION_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete the original-validation measurement and one-step weight-handoff check first.")
        expansion_result = run_expansion_segment(expansion, "continue", finllm)
        print(json.dumps(expansion_result, indent=2, ensure_ascii=False))
        EXPANSION_TRAIN_SUCCEEDED = True
        '''), code('''
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_TRAIN_SUCCEEDED", False):
            raise RuntimeError("New-stage training has not completed successfully; keep existing backups.")
        retention_after = retention_check(expansion, "after")
        print(json.dumps(retention_after, indent=2, ensure_ascii=False))
        EXPANSION_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        EXPANSION_TRAIN_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "EXPANSION_RETENTION_BEFORE_SUCCEEDED", "EXPANSION_PROBE_SUCCEEDED",
            "EXPANSION_TRAIN_SUCCEEDED", "EXPANSION_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("Measurement, one-step verification and training have not all succeeded; do not report a complete stage backup.")
        expansion_train_backup = backup_expansion_training(expansion)
        print(json.dumps(expansion_train_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", expansion["paths"]["drive"] / "runtime_revision.txt")
        EXPANSION_TRAIN_BACKED_UP = True
        print("Expanded general training and verified backup complete:", expansion["paths"]["drive"])
        print("Original seed data and 300-update artifacts preserved:", expansion["paths"]["seed_drive"])
        '''), markdown('''
        Keep one-step and training-segment reports, retention before/after, actual token counts and the final `verified: true` backup output.
        Model architecture and vocabulary are unchanged; record original seed 300 updates separately from new-stage updates.
        This notebook trains only on expanded general text; it does not automatically start financial-domain training, SFT or DPO.
        Notebook 01b prepares expanded financial shards. The next stage requires matching mixture configuration and entry points; human asset/stance labels require separate confirmation.
        ''')]
    write("02c_expanded_pretrain.ipynb", cells)


if __name__ == "__main__":
    build()
