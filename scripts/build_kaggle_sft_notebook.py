"""Generate the Kaggle training frontend for the user's trusted 74-step model."""
from build_notebooks import BOOTSTRAP, code, markdown, write


def build():
    bootstrap = BOOTSTRAP.replace('/content', '/kaggle/working')
    cells = [markdown('''
        # 04c | Single-GPU entity-stance SFT pilot on Kaggle

        Use the **private** transfer package and SHA256 from 04b to initialize from the 74-update domain_v1 weights.
        The first cross-platform handoff imports model weights and creates a new optimizer; it does not restore the Colab optimizer.
        Use Kaggle Add Input to attach a **private** dataset containing `130u_kaggle_transfer.zip`.
        In Session options, select GPU T4 x2 and Internet on. This pilot uses only the first GPU.

        Run cells in order and record the actual commit SHA from the first cell. If main changes before cross-session recovery,
        replace `REF = "main"` in the first cell with the original full 40-character SHA, then run cells in order.
        Recovery requires code, data, paths, PyTorch/CUDA, device and AMP settings that match the original run.
        Training results and temporary files are in
        `/kaggle/working`, not Google Drive; persistence after runtime termination is not guaranteed.
        Before ending the session, follow the output instructions to download the recovery ZIP or save a private notebook version with outputs.
        Do not publish uploaded data or outputs containing model weights or source headlines.

        This research pilot predicts bullish/neutral/bearish stance toward headline entities. Asset identity, semantic events
        and chronological isolation remain unaudited; development metrics do not measure security prediction or real X performance.
        '''), code(bootstrap), code('''
        from finllm.kaggle_sft_runtime import (install_transfer_archive, install_transfer_directory,
            install_recovery_archive, install_recovery_directory)
        from finllm.sft_pilot_runtime import prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
        from pathlib import Path

        # Copy the transfer manifest SHA256 and ZIP SHA256 separately from 04b output.
        TRANSFER_MANIFEST_SHA256 = "PASTE_TRANSFER_MANIFEST_SHA256_FROM_04B".strip().lower()
        TRANSFER_ZIP_SHA256 = "PASTE_TRANSFER_ZIP_SHA256_FROM_04B".strip().lower()
        # For cross-session recovery, attach the prior recovery ZIP as private Input and enter its SHA.
        # Leave empty on the first run or when rerunning this notebook within the same session.
        RECOVERY_MANIFEST_SHA256 = "".strip().lower()
        RECOVERY_ZIP_SHA256 = "".strip().lower()
        INPUT_ROOT = Path("/kaggle/input")
        HANDOFF = Path("/kaggle/working/130u_handoff")
        manifests = sorted(INPUT_ROOT.rglob("transfer_manifest.json"))
        archives = sorted(INPUT_ROOT.rglob("130u_kaggle_transfer.zip"))
        if len(manifests) == 1:
            transfer = install_transfer_directory(manifests[0].parent, HANDOFF,
                                                  expected_manifest_sha256=TRANSFER_MANIFEST_SHA256)
        elif not manifests and len(archives) == 1:
            transfer = install_transfer_archive(archives[0], HANDOFF, expected_zip_sha256=TRANSFER_ZIP_SHA256)
        else:
            raise RuntimeError("Use Add Input to attach exactly one private transfer dataset.")
        print("Original artifacts verified:", json.dumps(transfer, indent=2, ensure_ascii=False))
        if RECOVERY_MANIFEST_SHA256 or RECOVERY_ZIP_SHA256:
            manifests = sorted(INPUT_ROOT.rglob("recovery_manifest.json"))
            candidates = sorted(INPUT_ROOT.rglob("130u_sft_recovery_*.zip"))
            if len(manifests) == 1:
                restored = install_recovery_directory(manifests[0].parent, HANDOFF,
                                         expected_manifest_sha256=RECOVERY_MANIFEST_SHA256)
            elif not manifests and len(candidates) == 1:
                restored = install_recovery_archive(candidates[0], HANDOFF,
                                         expected_zip_sha256=RECOVERY_ZIP_SHA256)
            else:
                raise RuntimeError("Attach exactly one private recovery dataset.")
            print("Private SFT progress restored:", json.dumps(restored, indent=2, ensure_ascii=False))
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("Select GPU T4 in Session options and rerun from the first cell.")
        print("GPU count:", torch.cuda.device_count(), "Using:", torch.cuda.get_device_name(0))
        pilot = prepare_sft_pilot(PROJECT, HANDOFF, run_version="sft_kaggle_v1")
        print("Verified domain parent model and actual exclusion pools:", json.dumps(pilot["parent_audit"], indent=2, ensure_ascii=False))
        print("Task partitions:", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        '''), code('''
        from finllm.kaggle_sft_runtime import export_recovery
        from datetime import datetime, timezone
        SFT_PROBE_SUCCEEDED = False
        SFT_TRAIN_SUCCEEDED = False
        SFT_EVAL_SUCCEEDED = False
        probe = run_sft_pilot(pilot, "probe", finllm)
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        SFT_PROBE_SUCCEEDED = True
        recovery_file = Path("/kaggle/working") / ("130u_sft_recovery_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
        print("One-step recovery package (download or save a version for persistence):", json.dumps(export_recovery(HANDOFF, recovery_file), indent=2, ensure_ascii=False))
        '''), code('''
        SFT_TRAIN_SUCCEEDED = False
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_PROBE_SUCCEEDED", False):
            raise RuntimeError("Complete the one-step check first.")
        result = run_sft_pilot(pilot, "continue", finllm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        SFT_TRAIN_SUCCEEDED = True
        '''), code('''
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False):
            raise RuntimeError("Complete training first.")
        output = Path(pilot["config"]["output_dir"])
        development = pilot["task_local"] / "development.jsonl"
        for name, checkpoint in (("domain_before", Path(pilot["config"]["init_checkpoint"])),
                                 ("sft_best", output / "best.pt")):
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", pilot["config"]["tokenizer"],
                   "--input", development, "--output", output / (name + "_development.json"),
                   "--target-mode", "literal_entity")
            print(name, (output / (name + "_development.json")).read_text(encoding="utf-8"))
        SFT_EVAL_SUCCEEDED = True
        '''), code('''
        if not globals().get("SFT_TRAIN_SUCCEEDED", False) or not globals().get("SFT_EVAL_SUCCEEDED", False):
            raise RuntimeError("Complete training and the development-set comparison first.")
        backup = backup_sft_pilot(pilot)
        print("Local run-result backup:", json.dumps(backup, indent=2, ensure_ascii=False))
        recovery_file = Path("/kaggle/working") / ("130u_sft_recovery_final_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
        print("Final recovery package (download or save private version outputs):", json.dumps(export_recovery(HANDOFF, recovery_file), indent=2, ensure_ascii=False))
        '''), markdown('''
        Recovery and transfer packages contain private data and must stay in your own account. Keep the recovery-package SHA,
        the code SHA from the first cell, development results and backup output. If a Kaggle process or session terminates unexpectedly,
        local progress not downloaded or saved may be lost; original domain weights remain on Drive.
        ''')]
    write("04c_kaggle_sft.ipynb", cells)


if __name__ == "__main__":
    build()
