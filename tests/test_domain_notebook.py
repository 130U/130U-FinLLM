"""Execute published domain-handoff cells in two actual isolated kernels.

External services use local substitutes and a tiny CPU fixture replaces the T4
budget. Installation, audited parent loading, CLI training and saved-state
recovery are real. These tests do not establish a new cloud training result.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import torch

from finllm.checkpoint import load_checkpoint, sha256_file
from test_notebook_execution import (_cells, _cpu_config_cell, _isolated_python,
                                     _kernel, _local_repository, _offline)


def _inventory(directory):
    return {p.relative_to(directory).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def _domain_cells():
    cells = _cells("03a_domain_pretrain.ipynb")
    setup = next(source for source in cells if "domain = prepare_domain_training(" in source)
    setup = _cpu_config_cell(setup).replace("MIN_PARENT_UPDATES = 1220", "MIN_PARENT_UPDATES = 4")
    setup = setup.replace("TARGET_UPDATES = 74", "TARGET_UPDATES = 3")
    return dict(bootstrap=cells[0], setup=setup,
                before=next(source for source in cells if "domain_retention_before = retention_domain_check" in source),
                probe=next(source for source in cells if "domain_probe = run_domain_segment" in source),
                train=next(source for source in cells if "domain_result = run_domain_segment" in source),
                after=next(source for source in cells if "domain_retention_after = retention_domain_check" in source),
                backup=next(source for source in cells if "domain_backup = backup_domain_training" in source))


def test_domain_notebook_generator_matches_and_has_one_resolver_transaction():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/03a_domain_pretrain.ipynb"
    before = path.read_bytes()
    subprocess.run([sys.executable, "-X", "utf8", "scripts/build_domain_notebook.py"],
                   cwd=root, check=True, capture_output=True, text=True, encoding="utf-8")
    assert path.read_bytes() == before
    cells = _cells(path.name)
    transactions = []
    for source in cells:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "command":
                strings = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if "pip" in strings and "install" in strings:
                    transactions.append(strings)
    assert len(transactions) == 1 and "." in transactions[0]
    joined = "\n".join(cells)
    assert "activate_project_in_kernel()" in joined
    assert "check_fresh_process_import()" in joined
    assert "MIN_PARENT_UPDATES = 1220" in joined
    assert "PROBE_UPDATES = 1" in joined and "TARGET_UPDATES = 74" in joined
    assert "MOUNT_DRIVE = True" in joined
    assert "parent_payload = load_checkpoint(" in joined
    assert "fetch_expansion_corpora.py" not in joined and "build_expansion_artifacts(" not in joined
    notebook = json.loads(before)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_domain_notebook_two_kernels_audits_parent_and_recovers_completed_stage(tmp_path):
    from test_domain_runtime import _domain_fixture

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_root = tmp_path / "fixture"
        fixture_root.mkdir()
        original, drive = _domain_fixture(fixture_root)
    finally:
        torch.set_num_threads(old_threads)
    seed_before = _inventory(drive / "seeds/seed_v1")
    expansion_before = _inventory(drive / "expansions/expansion_v1")
    parent_file = drive / "expansions/expansion_v1/run/latest.pt"
    parent_hash = sha256_file(parent_file)
    parent_payload = load_checkpoint(parent_file)
    assert parent_payload["progress"]["updates"] == 4
    repository = _local_repository(tmp_path / "upstream")
    python = _isolated_python(tmp_path / "python")
    project = tmp_path / "first-runtime/130U-FinLLM"
    project.parent.mkdir()
    first = tmp_path / "first-kernel"
    first.mkdir()
    cells = _domain_cells()
    with _kernel(python, first) as execute:
        execute("import importlib.util\nassert importlib.util.find_spec('finllm') is None")
        execute(_offline(cells["bootstrap"], repository, project))
        execute("from finllm.domain_runtime import prepare_domain_training\nassert callable(prepare_domain_training)")
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        output = execute(cells["setup"])
        assert parent_hash in output
        execute("assert DOMAIN_INPUTS_VERIFIED is True\nassert domain['restored_updates'] == 0")
        execute(f"assert parent_audit['checkpoint_sha256'] == {parent_hash!r}\nassert parent_audit['updates'] == 4")
        assert not (project / "runs/domain_v1/latest.pt").exists()
        execute(cells["train"], expected_error="RuntimeError")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("saved_retention = retention_domain_check\ndef retention_domain_check(*args):\n    raise RuntimeError('simulated domain retention failure')")
        execute(cells["before"], expected_error="RuntimeError")
        execute("assert DOMAIN_RETENTION_BEFORE_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("retention_domain_check = saved_retention")
        execute(cells["before"])
        execute("saved_runner = finllm\ndef finllm(*args):\n    raise RuntimeError('simulated domain training failure')")
        execute(cells["probe"], expected_error="RuntimeError")
        execute("assert DOMAIN_PROBE_SUCCEEDED is False\nassert DOMAIN_TRAIN_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("finllm = saved_runner")
        execute(cells["probe"])
        execute("assert domain_probe['start_updates'] == 0\nassert domain_probe['end_updates'] == 1")
        run = project / "runs/domain_v1"
        first_checkpoint = load_checkpoint(run / "latest.pt")
        assert {float(s["step"]) for s in first_checkpoint["optimizer_state"]["state"].values()} == {1.0}
        assert {float(s["step"]) for s in parent_payload["optimizer_state"]["state"].values()} == {4.0}
        execute("def finllm(*args):\n    raise RuntimeError('simulated domain training failure')")
        execute(cells["train"], expected_error="RuntimeError")
        execute("assert DOMAIN_TRAIN_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("finllm = saved_runner")
        execute(cells["train"])
        execute("def retention_domain_check(*args):\n    raise RuntimeError('simulated domain retention failure')")
        execute(cells["after"], expected_error="RuntimeError")
        execute("assert DOMAIN_RETENTION_AFTER_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("retention_domain_check = saved_retention")
        execute(cells["after"])
        output = execute(cells["backup"])
        assert '"verified": true' in output
        execute("assert DOMAIN_BACKED_UP is True")
        execute("assert domain_result['start_updates'] == 1\nassert domain_result['end_updates'] == 3")
        execute("assert domain_result['model_path_total_tokens'] == domain_result['new_experiment_tokens'] + domain_result['parent_tokens'] + domain_result['seed_tokens']")
        summary_bytes = (run / "run_summary.json").read_bytes()
        summary = json.loads(summary_bytes)
        assert summary["updates"] == 3 and summary["elapsed_seconds"] > 0
        assert load_checkpoint(run / "latest.pt")["config"]["stage"] == "domain"
        frozen_parent = Path(execute("print(parent_checkpoint)").strip())
        assert sha256_file(frozen_parent) == parent_hash
    assert _inventory(drive / "seeds/seed_v1") == seed_before
    assert _inventory(drive / "expansions/expansion_v1") == expansion_before

    fresh_python = _isolated_python(tmp_path / "fresh-python")
    restored_project = tmp_path / "fresh-runtime/130U-FinLLM"
    restored_project.parent.mkdir()
    second = tmp_path / "second-kernel"
    second.mkdir()
    with _kernel(fresh_python, second) as execute:
        execute(_offline(cells["bootstrap"], repository, restored_project))
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(cells["setup"])
        execute("assert domain['restored_updates'] == 3")
        run = restored_project / "runs/domain_v1"
        before = _inventory(run)
        assert (run / "run_summary.json").read_bytes() == summary_bytes
        for name in ("before", "probe", "train", "after", "backup"):
            execute(cells[name])
        execute("assert domain_probe['new_updates'] == 0\nassert domain_result['new_updates'] == 0")
        after = _inventory(run)
        assert after.keys() == before.keys()
        for name in before:
            assert after[name][0] == before[name][0]
            if name not in {"report.json", "run_summary.json"}:
                assert after[name][1] == before[name][1]
        assert (run / "run_summary.json").read_bytes() == summary_bytes
        assert sha256_file(run / "latest.pt") == sha256_file(drive / "live_checkpoints/domain_v1/latest.pt")
    assert _inventory(drive / "seeds/seed_v1") == seed_before
    assert _inventory(drive / "expansions/expansion_v1") == expansion_before
