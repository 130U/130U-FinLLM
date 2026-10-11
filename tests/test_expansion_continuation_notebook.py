"""Run published 02d cells in two real kernels using an actual tiny CPU chain.

Only cloud services and the model budget are substituted. Editable installation,
checkpoint state, CLI execution, retention measurements and snapshot recovery
are real; this is not evidence of a Colab/T4 training run.
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


def _continuation_cells():
    cells = _cells("02d_expanded_continue.ipynb")
    setup = next(source for source in cells if "continuation = prepare_expansion_continuation(" in source)
    setup = _cpu_config_cell(setup).replace("COMPLETED_UPDATES = 610", "COMPLETED_UPDATES = 4")
    setup = setup.replace("PROBE_UPDATES = 611", "PROBE_UPDATES = 5")
    setup = setup.replace("TARGET_UPDATES = 1220", "TARGET_UPDATES = 7")
    setup = setup.replace("target_updates=TARGET_UPDATES)", "target_updates=TARGET_UPDATES, min_parent_updates=2)")
    return dict(bootstrap=cells[0], setup=setup,
                before=next(source for source in cells if "continuation_retention_before = retention_continuation_check" in source),
                probe=next(source for source in cells if "continuation_probe = run_expansion_continuation" in source),
                train=next(source for source in cells if "continuation_result = run_expansion_continuation" in source),
                after=next(source for source in cells if "continuation_retention_after = retention_continuation_check" in source),
                backup=next(source for source in cells if "continuation_backup = backup_expansion_continuation" in source))


def test_continuation_notebook_generator_matches_and_has_single_resolver_transaction():
    root = Path(__file__).resolve().parents[1]
    path = root / "notebooks/02d_expanded_continue.ipynb"
    before = path.read_bytes()
    subprocess.run([sys.executable, "-X", "utf8", "scripts/build_expanded_continue_notebook.py"],
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
    assert "COMPLETED_UPDATES = 610" in joined and "TARGET_UPDATES = 1220" in joined
    assert "MOUNT_DRIVE = True" in joined
    assert "build_expansion_artifacts(" not in joined and "fetch_expansion_corpora.py" not in joined
    notebook = json.loads(before)
    assert all(not cell.get("outputs") and cell.get("execution_count") is None
               for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_continuation_notebook_in_two_isolated_kernels_preserves_completed_evidence(tmp_path):
    from test_expansion_runtime import _expansion_fixture, _prepare, _finish
    from finllm.expansion_runtime import backup_expansion_training

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_root = tmp_path / "fixture"
        fixture_root.mkdir()
        original, drive = _expansion_fixture(fixture_root)
        predecessor = _prepare(original, drive)
        _finish(predecessor)
        assert backup_expansion_training(predecessor)["verified"]
    finally:
        torch.set_num_threads(old_threads)
    seed_before = _inventory(drive / "seeds/seed_v1")
    retention_before = _inventory(drive / "expansions/expansion_v1/retention")
    predecessor_summary = (drive / "expansions/expansion_v1/run/run_summary.json").read_bytes()
    predecessor_checkpoint_hash = sha256_file(drive / "expansions/expansion_v1/run/latest.pt")
    repository = _local_repository(tmp_path / "upstream")
    python = _isolated_python(tmp_path / "python")
    project = tmp_path / "first-runtime/130U-FinLLM"
    project.parent.mkdir()
    first = tmp_path / "first-kernel"
    first.mkdir()
    cells = _continuation_cells()
    with _kernel(python, first) as execute:
        execute("import importlib.util\nassert importlib.util.find_spec('finllm') is None")
        execute(_offline(cells["bootstrap"], repository, project))
        execute("from finllm.expansion_continuation import prepare_expansion_continuation\nassert callable(prepare_expansion_continuation)")
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(cells["setup"])
        execute("assert continuation['restored_updates'] == 4")
        execute(cells["train"], expected_error="RuntimeError")
        execute(cells["backup"], expected_error="RuntimeError")
        execute(cells["before"])
        execute("saved_runner = finllm\ndef finllm(*args):\n    raise RuntimeError('simulated continuation failure')")
        execute(cells["probe"], expected_error="RuntimeError")
        execute("assert CONTINUATION_PROBE_SUCCEEDED is False\nassert CONTINUATION_TRAIN_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("finllm = saved_runner")
        execute(cells["probe"])
        execute("assert continuation_probe['start_updates'] == 4\nassert continuation_probe['end_updates'] == 5")
        execute("def finllm(*args):\n    raise RuntimeError('simulated continuation failure')")
        execute(cells["train"], expected_error="RuntimeError")
        execute("assert CONTINUATION_TRAIN_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("finllm = saved_runner")
        # A retention error after training cannot authorize a completed backup.
        execute(cells["train"])
        execute("saved_retention = retention_continuation_check\ndef retention_continuation_check(*args):\n    raise RuntimeError('simulated retention failure')")
        execute(cells["after"], expected_error="RuntimeError")
        execute("assert CONTINUATION_RETENTION_AFTER_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("retention_continuation_check = saved_retention")
        execute(cells["after"])
        output = execute(cells["backup"])
        assert '"verified": true' in output
        execute("assert CONTINUATION_BACKED_UP is True")
        run = project / "runs/expansion_v1_pretrain"
        summary = json.loads((run / "run_summary.json").read_text())
        assert summary["updates"] == 7
        assert load_checkpoint(run / "latest.pt")["progress"]["updates"] == 7
        execute("assert continuation_result['segment_updates'] == 3\nassert continuation_result['new_experiment_tokens'] == 448")
        execute("assert (Path(continuation['predecessor_archive']) / 'run_summary.json').read_bytes() == " + repr(predecessor_summary))
        archive = Path(execute("print(continuation['predecessor_archive'])").strip())
        assert sha256_file(archive / "latest.pt") == predecessor_checkpoint_hash
    assert _inventory(drive / "seeds/seed_v1") == seed_before
    assert _inventory(drive / "expansions/expansion_v1/retention") == retention_before

    fresh_python = _isolated_python(tmp_path / "fresh-python")
    restored_project = tmp_path / "fresh-runtime/130U-FinLLM"
    restored_project.parent.mkdir()
    second = tmp_path / "second-kernel"
    second.mkdir()
    with _kernel(fresh_python, second) as execute:
        execute(_offline(cells["bootstrap"], repository, restored_project))
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(cells["setup"])
        execute("assert continuation['restored_updates'] == 7")
        run = restored_project / "runs/expansion_v1_pretrain"
        before = _inventory(run)
        for name in ("before", "probe", "train", "after", "backup"):
            execute(cells[name])
        execute("assert continuation_probe['new_updates'] == 0\nassert continuation_result['new_updates'] == 0")
        after = _inventory(run)
        assert after.keys() == before.keys()
        for name in before:
            assert after[name][0] == before[name][0]
            if name not in {"report.json", "run_summary.json"}:
                assert after[name][1] == before[name][1]
        assert sha256_file(run / "latest.pt") == sha256_file(drive / "live_checkpoints/expansion_v1_pretrain/latest.pt")
        assert json.loads((run / "run_summary.json").read_text())["elapsed_seconds"] > 0
    assert _inventory(drive / "seeds/seed_v1") == seed_before
    assert _inventory(drive / "expansions/expansion_v1/retention") == retention_before
