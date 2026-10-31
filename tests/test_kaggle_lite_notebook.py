"""Exercise actual Kaggle notebook cells against a tiny real transferred model."""
import json
import zipfile
from pathlib import Path

import torch

import finllm.kaggle_lite as lite
from test_kaggle_lite import mini_project


def test_lite_notebook_cells_install_train_evaluate_and_resume(mini_project, tmp_path, capsys):
    project, drive, domain_hash, tokenizer_hash, task_hash = mini_project
    archive = tmp_path / "130u_kaggle_lite.zip"
    report = lite.export_lite_from_drive(drive, archive, expected_domain_sha256=domain_hash,
        expected_tokenizer_sha256=tokenizer_hash, expected_task_sha256=task_hash,
        expected_domain_updates=3, expected_task_rows=50)
    private_input, working = tmp_path / "kaggle-input", tmp_path / "kaggle-working"
    private_input.mkdir()
    working.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(private_input / "private-dataset")
    notebook = json.loads((Path(__file__).resolve().parents[1] / "notebooks/04e_kaggle_lite_sft.ipynb").read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    # Execute the real input, training and evaluation cells; bootstrap/network
    # and GPU discovery are the only environment-specific pieces replaced.
    namespace = {"PROJECT": project, "Path": Path, "json": json, "__name__": "__main__"}
    env_cell = (code[1].replace('"PASTE_LITE_MANIFEST_SHA256"', json.dumps(report["manifest_sha256"]))
                       .replace('"/kaggle/input"', json.dumps(str(private_input)))
                       .replace('"/kaggle/working/', json.dumps(str(working) + "/")[:-1]))
    train_cell = (code[2].replace("if not torch.cuda.is_available():", "if False:")
                        .replace("torch.cuda.get_device_name(0)", '"CPU fixture"'))
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        exec(compile(env_cell, "04e-input", "exec"), namespace)
        exec(compile(train_cell, "04e-train", "exec"), namespace)
        exec(compile(code[3], "04e-eval", "exec"), namespace)
        result = namespace["result"]
        assert result["updates"] == 2
        assert (working / "130u_lite_sft_v1/sft_best_development.json").is_file()
        assert "Recovery checkpoint SHA256" in capsys.readouterr().out
        exec(compile(train_cell, "04e-resume", "exec"), namespace)
        assert namespace["result"]["updates"] == 2
    finally:
        torch.set_num_threads(old)
