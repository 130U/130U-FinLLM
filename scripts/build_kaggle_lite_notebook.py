"""Generate the private Kaggle frontend for an existing lite weights handoff."""
from build_notebooks import BOOTSTRAP, code, markdown, write


def build():
    bootstrap = BOOTSTRAP.replace('/content', '/kaggle/working')
    cells = [markdown('''
        # 04e | Lightweight financial-model handoff and SFT on Kaggle

        Use the private lightweight package exported from Colab, with trained `domain_v1` weights and the original tokenizer.
        The complete 1.54 GiB transfer package is unnecessary. The first handoff imports only model weights and creates a new optimizer;
        it is not exact cross-platform optimizer recovery. Model and label packages are private; use Private datasets and notebooks.

        Attach the private lightweight dataset on Kaggle and select GPU T4 x2 with Internet on.
        Only the first GPU is used. Run cells in order and enter the package SHA256 in the configuration cell.
        The package retains source-verification reports from the Colab export, but Kaggle cannot reread original manifests that were not transferred.
        Results concern exploratory headline literal-entity stance, not security prediction or X validation.
        '''), code(bootstrap), code('''
        from finllm.kaggle_lite import install_lite_archive, install_lite_directory, prepare_lite_sft
        from finllm.training import run_training
        from finllm.checkpoint import load_checkpoint, sha256_file
        from finllm.runtime import restore_seed_checkpoint
        from finllm.evaluation import run_evaluation
        import torch

        # Copy the manifest SHA256 from the lightweight export report; use the same inputs and code revision for recovery.
        LITE_MANIFEST_SHA256 = "PASTE_LITE_MANIFEST_SHA256".strip().lower()
        # Enter the ZIP SHA256 only if Kaggle did not extract the ZIP automatically.
        LITE_ZIP_SHA256 = "".strip().lower()
        INPUT = Path("/kaggle/input")
        LITE = Path("/kaggle/working/130u_lite_input")
        RUN = Path("/kaggle/working/130u_lite_sft_v1")
        manifests = sorted(INPUT.rglob("lite_manifest.json"))
        archives = sorted(INPUT.rglob("130u_kaggle_lite.zip"))
        if len(manifests) == 1:
            installed = install_lite_directory(manifests[0].parent, LITE,
                                              expected_manifest_sha256=LITE_MANIFEST_SHA256)
        elif not manifests and len(archives) == 1:
            installed = install_lite_archive(archives[0], LITE, expected_zip_sha256=LITE_ZIP_SHA256)
        else:
            raise RuntimeError("Use Add Input to attach exactly one Private lightweight dataset.")
        print("Lightweight input verification:", json.dumps(installed, indent=2, ensure_ascii=False))
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("Select T4 GPU and rerun from the first cell.")
        print("GPU count:", torch.cuda.device_count(), "Using:", torch.cuda.get_device_name(0))
        pilot = prepare_lite_sft(PROJECT, LITE, RUN,
                                 expected_manifest_sha256=installed["manifest_sha256"], epochs=2)
        config = pilot["config"]
        print("Task partitions:", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        latest = restore_seed_checkpoint(RUN, Path(config["backup_dir"]))
        if latest:
            previous = load_checkpoint(latest, allow_previous=False)
            if previous.get("config", {}).get("lite_binding") != config.get("lite_binding"):
                raise ValueError("The existing checkpoint and lightweight input identities differ; preserve the original experiment and stop.")
            config.pop("init_checkpoint", None)
            config["resume"] = str(latest)
            print("Existing checkpoint detected; restore strictly with the same runtime environment.")
        result = run_training(config)
        print("SFT training results:", json.dumps(result, indent=2, ensure_ascii=False))
        '''), code('''
        for name, checkpoint in (("domain_before", LITE / "weights/domain_weights.pt"),
                                 ("sft_best", RUN / "best.pt")):
            metrics = run_evaluation(checkpoint, config["tokenizer"], config["validation_data"],
                                     RUN / (name + "_development.json"), target_mode="literal_entity")
            print(name, "Development results (exploratory, not final test):", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("Model and training state:", RUN)
        print("Tokenizer and transfer inputs:", LITE)
        print("Recovery checkpoint SHA256:", sha256_file(RUN / "latest.pt"))
        print("Recovery integrity marker SHA256:", sha256_file(RUN / "latest.pt.json"))
        print("Save a Private notebook version with outputs, or download latest.pt and latest.pt.json;")
        print("also preserve the Private lightweight dataset and code revision from the first cell. Working-directory persistence is not guaranteed.")
        '''), markdown('''
        After disconnection, rerun from the first cell if files remain in the same session. Cross-session recovery requires preserving and restoring the original working directory,
        pinning the first-cell `REF` to the original full 40-character commit, with matching device, PyTorch/CUDA, data and paths.
        Original Drive weights remain unchanged. Development metrics are not final evaluation; original publisher rights require separate verification.
        ''')]
    write("04e_kaggle_lite_sft.ipynb", cells)


if __name__ == "__main__":
    build()
