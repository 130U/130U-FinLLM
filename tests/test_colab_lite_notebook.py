"""Exercise the actual Colab CPU export cell against a genuine tiny model."""
import json
from pathlib import Path
import zipfile

import finllm.kaggle_lite as lite
from finllm.checkpoint import sha256_file
from test_kaggle_lite import mini_project


def test_colab_lite_export_cell(mini_project, tmp_path, monkeypatch, capsys):
    _, drive, domain_hash, tokenizer_hash, task_hash = mini_project
    real_export = lite.export_lite_from_drive

    def small_export(root, output):
        return real_export(root, output, expected_domain_sha256=domain_hash,
                           expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
                           expected_domain_updates=3, expected_task_rows=50)

    monkeypatch.setattr(lite, "export_lite_from_drive", small_export)
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/04d_colab_lite_export.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    output = tmp_path / "130u_kaggle_lite.zip"
    cell = code[2].replace('"/content/130u_kaggle_lite.zip"', json.dumps(str(output)))
    namespace = {"DRIVE_ROOT": drive, "Path": Path, "json": json, "__name__": "__main__"}
    exec(compile(cell, "04d-export", "exec"), namespace)
    report = namespace["lite_report"]
    assert output.is_file() and report["sha256"] == sha256_file(output)
    with zipfile.ZipFile(output) as archive:
        assert "lite_manifest.json" in archive.namelist()
        assert "weights/domain_weights.pt" in archive.namelist()
    output_text = capsys.readouterr().out
    assert report["sha256"] in output_text and report["manifest_sha256"] in output_text
