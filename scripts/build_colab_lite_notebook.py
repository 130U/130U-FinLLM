"""Generate a small, CPU-only transfer of the user's already trained weights."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04d | Export a lightweight Kaggle model package on CPU

        Extract model weights from the verified 74-update `domain_v1` checkpoint on Drive, together with the original 8K tokenizer and reviewed task data.
        The old 1.54 GiB transfer package is unnecessary; Drive originals stay unchanged. Select CPU and run from the first cell; this notebook performs no training.
        Verify the domain checkpoint SHA, tokenizer and task manifest before downloading the roughly 100 MB package; consult the output for actual size.

        The earlier 04b cloud report confirmed that four exclusion pools cover all 1,008 headlines. This lightweight package preserves that report,
        but does not carry all historical corpora, so Kaggle cannot independently recheck those exclusion pools.
        It contains private model weights and news headlines; store it only in your own Kaggle Private dataset.
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.kaggle_lite import export_lite_from_drive
        LITE_ZIP = Path("/content/130u_kaggle_lite.zip")
        lite_report = export_lite_from_drive(DRIVE_ROOT, LITE_ZIP)
        print("Lightweight model package created:", json.dumps(lite_report, indent=2, ensure_ascii=False))
        print("Record ZIP SHA256:", lite_report["sha256"])
        print("Record manifest_sha256:", lite_report["manifest_sha256"])
        '''), code('''
        from google.colab import files
        if not LITE_ZIP.is_file():
            raise FileNotFoundError("Generate the lightweight model package first.")
        files.download(str(LITE_ZIP))
        print("Browser download started; wait for completion before uploading to a Kaggle Private dataset.")
        '''), markdown('''
        The transfer retains model weights but not the old optimizer state. Kaggle starts SFT with a new optimizer.
        Before rerunning, confirm the downloaded file is safe and use a new filename; leave original Drive artifacts unchanged.
        ''')]
    write("04d_colab_lite_export.ipynb", cells)


if __name__ == "__main__":
    build()
