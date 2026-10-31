"""Generate a CPU-only, user-controlled Colab-to-Kaggle transfer notebook."""
from build_notebooks import BOOTSTRAP, code, markdown, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04b | Prepare a private Kaggle transfer package on Colab CPU

        Read your Drive snapshots and validate the domain checkpoint, 8K vocabulary and actual pretraining exclusion manifests,
        then stream the required verified source snapshots into a ZIP. No GPU or training is required, and the Drive model is unchanged.

        In Colab, select Runtime > Change runtime type > None, run cells in order and mount your own Drive.
        Task packages, user annotations and models are private; upload them only as a Kaggle **Private** dataset.
        The transfer package is stored in temporary `/content`; download it before disconnecting. Actual size is reported and may exceed 1 GB.
        Record ZIP SHA256 and manifest_sha256 when complete. Do not run another training process or modify source snapshots concurrently.
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.kaggle_export import export_kaggle_bundle
        TRANSFER_ZIP = Path("/content/130u_kaggle_transfer.zip")
        export_report = export_kaggle_bundle(DRIVE_ROOT, TRANSFER_ZIP)
        print("Private Kaggle transfer package verified:", json.dumps(export_report, indent=2, ensure_ascii=False))
        print("Record ZIP SHA256:", export_report["sha256"])
        print("Record manifest_sha256 (used after extraction on Kaggle):", export_report["manifest_sha256"])
        '''), code('''
        from google.colab import files
        if not TRANSFER_ZIP.is_file():
            raise FileNotFoundError("Generate and verify the transfer package in the previous cell first.")
        files.download(str(TRANSFER_ZIP))
        print("Browser download started. Wait for completion before uploading to a private Kaggle dataset.")
        '''), markdown('''
        This notebook does not start SFT or confirm receipt by Kaggle. Preserve complete errors if validation fails or disk space is insufficient.
        If a temporary file with the same name remains after interruption, choose a new ZIP filename; do not overwrite or delete Drive originals.
        ''')]
    write("04b_kaggle_export.ipynb", cells)


if __name__ == "__main__":
    build()
