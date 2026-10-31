"""Build the bounded real-data path added after the initial smoke run."""
from build_notebooks import markdown, code, start, write


SEED_BACKUP = '''
SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
for name, source in (("raw", Path("data/raw") / SEED_VERSION),
                     ("processed", Path("data/processed") / SEED_VERSION),
                     ("tokenizer", Path("tokenizer") / SEED_VERSION),
                     ("run", Path("runs") / (SEED_VERSION + "_pretrain"))):
    if source.exists():
        finllm("sync", "--source", PROJECT / source, "--destination", SEED_DRIVE / name)
shutil.copy2(PROJECT / "runtime_revision.txt", SEED_DRIVE / "runtime_revision.txt")
print("Version backup complete:", SEED_DRIVE)
'''


def build():
    cells = start("01a | Initial real corpus and 8K vocabulary", "Download bounded, pinned Wikipedia and SEntFiN candidate headlines, and save original FOMC minutes with dates and hashes. Freeze all candidate task texts before training the tokenizer and packing shards. This notebook neither converts news sentiment automatically into bullish/bearish stance nor starts model training. Raw text should total tens of MB; download libraries use additional cache space.")
    cells[0]["source"] = [line.replace("Colab menu: Runtime > Change runtime type > T4 GPU.",
        "CPU is sufficient for corpus preparation here; switch to T4 GPU before training in 02a.") for line in cells[0]["source"]]
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        print("Drive project directory:", DRIVE_ROOT)
        ''')
    cells[3] = markdown("This notebook defaults to seed_v1. Reruns validate and reuse complete artifacts; new runtimes restore this version from Drive. Missing or modified artifacts trigger explicit errors and are preserved; choose a new version or restore a complete backup. Raw material stays in the runtime and your own Drive, not GitHub.")
    cells += [code('''
        SEED_VERSION = "seed_v1"
        ROOT = Path("data/raw") / SEED_VERSION
        CORPORA = ROOT / "corpora"
        TASK = ROOT / "task"
        PREPARED = ROOT / "prepared"
        TOKENIZER = Path("tokenizer") / SEED_VERSION
        PROCESSED = Path("data/processed") / SEED_VERSION
        RUN_OUTPUT = Path("runs") / (SEED_VERSION + "_pretrain")
        command(sys.executable, "-m", "pip", "install", "-q", "-e", ".[data]", cwd=PROJECT)
        SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
        for name, destination in (("raw", ROOT), ("processed", PROCESSED), ("tokenizer", TOKENIZER)):
            source = SEED_DRIVE / name
            if not destination.exists() and source.exists():
                require(source / "SNAPSHOT_COMPLETE.json")
                finllm("sync", "--source", source, "--destination", PROJECT / destination, "--working-copy")
        SEED_BUILD_ARGS = ["--corpora", CORPORA, "--task-pool", TASK / "task_exclusions.jsonl",
                           "--prepared", PREPARED, "--tokenizer", TOKENIZER,
                           "--processed", PROCESSED, "--run-output", RUN_OUTPUT]
        print("Data preparation only; CUDA available:", torch.cuda.is_available())
        '''), code('''
        command(sys.executable, "-u", "scripts/fetch_seed_corpora.py", "--output", CORPORA,
                "--general-max-bytes", "20000000", cwd=PROJECT)
        command(sys.executable, "-u", "scripts/prepare_sentfin_review.py", "--download", "--output", TASK, cwd=PROJECT)
        '''), code('''
        command(sys.executable, "-u", "scripts/build_seed_artifacts.py", *SEED_BUILD_ARGS, cwd=PROJECT)
        print((PROCESSED / "seed_summary.json").read_text(encoding="utf-8"))
        '''), code('# Recheck complete artifacts before backup; interrupted partial outputs must not be reported as successful.\n'
                   'command(sys.executable, "-u", "scripts/build_seed_artifacts.py", *SEED_BUILD_ARGS, '
                   '"--validate-only", cwd=PROJECT)\n' + SEED_BACKUP),
        markdown("Inspect actual token counts in seed_summary.json after completion. The 200-row review sheet in data/raw/seed_v1/task records your assessment of target-asset stance; original human sentiment labels remain in a separate column. Notebook 02a uses these artifacts for an approximately 1M-token 24M pretraining pilot.")]
    write("01a_seed_data.ipynb", cells)

    cells = start("02a | 24M pretraining pilot on the initial real corpus", "Start from random initialization with the seed_v1 8K tokenizer and Wikipedia shards for 61 updates (about 1M tokens). Complete 01a first. This notebook runs only a short general-pretraining segment; it does not automatically continue to 100M tokens, SFT or DPO.")
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        ''')
    cells[3] = code('''
        SEED_VERSION = "seed_v1"
        SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
        # Separate version snapshots do not overwrite backup manifests for other versions or small experiments.
        for name, relative, required in (
            ("processed", f"data/processed/{SEED_VERSION}", "seed_summary.json"),
            ("tokenizer", f"tokenizer/{SEED_VERSION}", "tokenizer.json")):
            destination = PROJECT / relative
            if not (destination / required).is_file():
                require(SEED_DRIVE / name / "SNAPSHOT_COMPLETE.json")
                finllm("sync", "--source", SEED_DRIVE / name,
                       "--destination", destination, "--working-copy")
        ''')
    cells += [code('''
        SEED_TRAIN_SUCCEEDED = False
        if not torch.cuda.is_available():
            raise RuntimeError("Select a T4 GPU runtime.")
        CONFIG = Path("data/processed") / SEED_VERSION / "pilot_config.json"
        require(CONFIG)
        runtime_config = json.loads(CONFIG.read_text(encoding="utf-8"))
        runtime_config["backup_dir"] = str(DRIVE_ROOT / "live_checkpoints" / (SEED_VERSION + "_pretrain"))
        runtime_path = Path("configs/runtime") / (SEED_VERSION + "_pilot.json")
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
        print(json.dumps(runtime_config, indent=2))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown("Run the next cell to start the short segment. Existing checkpoints for this version are validated and resumed automatically; incompatible data or runtime settings raise errors. If the segment budget is already reached, results are read without changing checkpoints or performance records."), code('''
        SEED_TRAIN_SUCCEEDED = False
        from finllm.runtime import restore_seed_checkpoint
        RESUME_CHECKPOINT = restore_seed_checkpoint(runtime_config["output_dir"], runtime_config["backup_dir"])
        arguments = ["train", "--config", runtime_path]
        if RESUME_CHECKPOINT:
            arguments += ["--resume", RESUME_CHECKPOINT]
        finllm(*arguments)
        RUN_SUMMARY = Path(runtime_config["output_dir"]) / "run_summary.json"
        require(RUN_SUMMARY, Path(runtime_config["output_dir"]) / "latest.pt",
                Path(runtime_config["output_dir"]) / "latest.pt.json")
        print(RUN_SUMMARY.read_text(encoding="utf-8"))
        SEED_TRAIN_SUCCEEDED = True
        '''), code('if not globals().get("SEED_TRAIN_SUCCEEDED", False):\n'
                   '    raise RuntimeError("The training cell has not completed successfully; resolve the error above first. Existing backups remain unchanged.")\n'
                   + SEED_BACKUP),
        markdown("Review the displayed GPU model and the peak memory, tokens/second, tokens and validation loss in run_summary.json before choosing the next budget. SEntFiN stance review and asset mapping remain prerequisites for SFT/DPO; this run does not establish financial prediction ability.")]
    write("02a_seed_pilot.ipynb", cells)

    cells = start("02b | Preserve the pilot and resume to 300 cumulative updates", "Complete 02a first. Restore an existing seed_v1 checkpoint, verify recovery from update 61 to 62, then continue to 300 cumulative updates (nominally about 4,915,200 tokens). Model, data, optimizer, learning-rate schedule and precision remain unchanged; only the stopping budget increases. Archive the original 61-update pilot report separately on Drive; do not reinitialize the model.")
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        ''')
    cells[3] = code('''
        CONTINUATION_SUCCEEDED = False
        RESUME_PROBE_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        PILOT_UPDATES = 61
        PROBE_UPDATES = 62
        TARGET_UPDATES = 300
        if not torch.cuda.is_available():
            raise RuntimeError("Select a T4 GPU runtime; a T4 checkpoint cannot be resumed on CPU.")
        from finllm.continuation import prepare_seed_continuation, run_seed_segment, backup_seed_continuation
        continuation = prepare_seed_continuation(
            PROJECT, DRIVE_ROOT, SEED_VERSION, pilot_updates=PILOT_UPDATES,
            probe_updates=PROBE_UPDATES, target_updates=TARGET_UPDATES)
        print("Restored update count:", continuation["restored_updates"])
        print("Original pilot report verified and archived:", continuation["pilot_archive"])
        print("GPU:", torch.cuda.get_device_name(0))
        ''')
    cells += [markdown("Verify the shortest resume segment first. If update 62 or later is already complete, this cell validates and reads existing progress without rolling back or rerunning. A mismatch in the Colab PyTorch/CUDA environment stops execution; preserve the error rather than changing precision, batch size or initialization to bypass validation."), code('''
        RESUME_PROBE_SUCCEEDED = False
        CONTINUATION_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        probe_result = run_seed_segment(continuation, "probe", finllm)
        print(json.dumps(probe_result, indent=2, ensure_ascii=False))
        RESUME_PROBE_SUCCEEDED = True
        '''), markdown("Continue to 300 cumulative updates, including the original 61. Keep pilot_config.json unchanged. Checkpoints back up automatically every 10 updates; after interruption, rerun from the first cell to restore saved progress. Reported updates/tokens are cumulative, while elapsed_seconds/tokens_per_second cover only new training in that call."), code('''
        CONTINUATION_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("RESUME_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete the resume verification above successfully first.")
        continuation_result = run_seed_segment(continuation, "continue", finllm)
        print(json.dumps(continuation_result, indent=2, ensure_ascii=False))
        CONTINUATION_SUCCEEDED = True
        '''), code('''
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_SUCCEEDED", False):
            raise RuntimeError("Continued training has not completed successfully; keep existing backups and resolve the error above.")
        backup_result = backup_seed_continuation(continuation)
        print(json.dumps(backup_result, indent=2, ensure_ascii=False))
        CONTINUATION_BACKED_UP = True
        print("Cumulative training and verified backup complete:", continuation["seed_drive"])
        print("Original pilot report:", continuation["pilot_archive"])
        '''), markdown("Keep the continuation reports and verified: true backup output. Compare validation loss and resource use before choosing the next corpus and budget. This notebook does not automatically start domain adaptation, SFT or DPO; keep the original 01a/02a configurations and seed_v1 backups.")]
    write("02b_seed_continue.ipynb", cells)


if __name__ == "__main__":
    build()
