"""Execute generated expansion cells; external services use local fixtures."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil

import pytest
import torch

from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.runtime import sync_snapshot
from test_notebook_execution import (_cells, _cpu_config_cell, _isolated_python,
                                     _kernel, _local_repository, _offline)


def _inventory(directory):
    return {p.relative_to(directory).as_posix(): (sha256_file(p), p.stat().st_mtime_ns)
            for p in directory.rglob("*") if p.is_file()}


def _training_cells():
    cells = _cells("02c_expanded_pretrain.ipynb")
    setup = next(source for source in cells if "expansion = prepare_expansion_training(" in source)
    setup = _cpu_config_cell(setup).replace("MIN_PARENT_UPDATES = 300", "MIN_PARENT_UPDATES = 2")
    setup = setup.replace("TARGET_UPDATES = 610", "TARGET_UPDATES = 3")
    return dict(bootstrap=cells[0], setup=setup,
                before=next(source for source in cells if "retention_before = retention_check" in source),
                probe=next(source for source in cells if "expansion_probe = run_expansion_segment" in source),
                train=next(source for source in cells if "expansion_result = run_expansion_segment" in source),
                after=next(source for source in cells if "retention_after = retention_check" in source),
                backup=next(source for source in cells if "expansion_train_backup = backup_expansion_training" in source))


def test_data_notebook_has_one_resolver_transaction_and_no_vocabulary_retraining():
    cells = _cells("01b_expand_data.ipynb")
    transactions = []
    for source in cells:
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "command":
                strings = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if "pip" in strings and "install" in strings:
                    transactions.append(strings)
    assert len(transactions) == 1 and ".[data]" in transactions[0]
    assert "train-tokenizer" not in "\n".join(cells)


def test_training_notebook_cells_in_two_live_kernels(tmp_path, monkeypatch):
    from test_expansion_runtime import _expansion_fixture

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        fixture_root = tmp_path / "fixture"
        fixture_root.mkdir()
        original, drive = _expansion_fixture(fixture_root, monkeypatch)
    finally:
        torch.set_num_threads(old_threads)
    seed_before = _inventory(drive / "seeds/seed_v1")
    repository = _local_repository(tmp_path / "upstream")
    python = _isolated_python(tmp_path / "python")
    project = tmp_path / "first-runtime/130U-FinLLM"
    project.parent.mkdir()
    first = tmp_path / "first-kernel"
    first.mkdir()
    cells = _training_cells()
    with _kernel(python, first) as execute:
        execute(_offline(cells["bootstrap"], repository, project))
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(cells["setup"])
        execute(cells["train"], expected_error="RuntimeError")
        execute(cells["backup"], expected_error="RuntimeError")
        execute(cells["before"])
        # A failed CLI cannot leave stale success flags or allow final backup.
        execute("saved_runner = finllm\ndef finllm(*args):\n    raise RuntimeError('simulated training failure')")
        execute(cells["probe"], expected_error="RuntimeError")
        execute("assert EXPANSION_PROBE_SUCCEEDED is False\nassert EXPANSION_TRAIN_SUCCEEDED is False")
        execute(cells["backup"], expected_error="RuntimeError")
        execute("finllm = saved_runner")
        execute(cells["probe"])
        execute(cells["train"])
        execute(cells["after"])
        output = execute(cells["backup"])
        assert '"verified": true' in output
        execute("assert EXPANSION_TRAIN_BACKED_UP is True")
        summary = json.loads((project / "runs/expansion_v1_pretrain/run_summary.json").read_text())
        assert summary["updates"] == 3
        assert load_checkpoint(project / "runs/expansion_v1_pretrain/latest.pt")["progress"]["updates"] == 3
    assert _inventory(drive / "seeds/seed_v1") == seed_before

    fresh_python = _isolated_python(tmp_path / "fresh-python")
    restored_project = tmp_path / "fresh-runtime/130U-FinLLM"
    restored_project.parent.mkdir()
    second = tmp_path / "second-kernel"
    second.mkdir()
    with _kernel(fresh_python, second) as execute:
        execute(_offline(cells["bootstrap"], repository, restored_project))
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(cells["setup"])
        execute("assert expansion['restored_updates'] == 3")
        run = restored_project / "runs/expansion_v1_pretrain"
        before = _inventory(run)
        for name in ("before", "probe", "train", "after", "backup"):
            execute(cells[name])
        execute("assert expansion_probe['new_updates'] == 0\nassert expansion_result['new_updates'] == 0")
        after = _inventory(run)
        assert after.keys() == before.keys()
        for name in before:
            # Strict CLI validation may write a no-op report before the runtime
            # restores the original measured bytes; checkpoint mtimes stay fixed.
            assert after[name][0] == before[name][0]
            if name not in {"report.json", "run_summary.json"}:
                assert after[name][1] == before[name][1]
        assert sha256_file(run / "latest.pt") == sha256_file(drive / "live_checkpoints/expansion_v1_pretrain/latest.pt")
    assert _inventory(drive / "seeds/seed_v1") == seed_before


def test_data_notebook_build_and_backup_cells_in_live_kernel(tmp_path, monkeypatch):
    from test_expansion_data import build_seed_expansion_fixture

    fixture_root = tmp_path / "inputs"
    fixture_root.mkdir()
    root = build_seed_expansion_fixture(fixture_root)
    drive = tmp_path / "drive"
    seed_raw = tmp_path / "seed-raw"
    shutil.copytree(root / "seed_prepared", seed_raw / "prepared")
    for name, source in (("raw", seed_raw), ("processed", root / "seed_processed"), ("tokenizer", root / "tokenizer")):
        sync_snapshot(source, drive / "seeds/seed_v1" / name)
    seed_before = _inventory(drive / "seeds/seed_v1")
    repository = _local_repository(tmp_path / "upstream")
    python = _isolated_python(tmp_path / "python")
    project = tmp_path / "runtime/130U-FinLLM"
    project.parent.mkdir()
    cells = _cells("01b_expand_data.ipynb")
    setup = next(source for source in cells if "expansion_paths = restore_expansion_materials" in source)
    fetch = next(source for source in cells if '"scripts/fetch_expansion_corpora.py"' in source)
    backup_source = next(source for source in cells if "source_backup = sync_snapshot" in source)
    build = next(source for source in cells if "expansion_data_result = build_expansion_artifacts" in source)
    backup = next(source for source in cells if "expansion_backup = backup_expansion_data" in source)

    # Offline installation still requests the real data extra in one transaction;
    # the large dependencies are reused and the editable package is installed.
    tree = ast.parse(_offline(cells[0], repository, project))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "command":
            strings = [a.value for a in node.args if isinstance(a, ast.Constant)]
            if "pip" in strings and "install" in strings:
                for value in node.args:
                    if isinstance(value, ast.Constant) and value.value == ".":
                        value.value = ".[data]"
    bootstrap = ast.unparse(ast.fix_missing_locations(tree))
    kernel_root = tmp_path / "data-kernel"
    kernel_root.mkdir()
    with _kernel(python, kernel_root) as execute:
        execute(bootstrap)
        execute("import datasets, requests, bs4\nimport finllm.expansion\nassert callable(finllm.expansion.build_expansion_artifacts)")
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(setup)
        execute(backup_source, expected_error="RuntimeError")
        execute(backup, expected_error="RuntimeError")
        execute(f'''real_command = command
def command(*args, cwd=None):
    if "scripts/fetch_expansion_corpora.py" in args:
        destination = Path(args[args.index("--output") + 1])
        if not destination.exists():
            shutil.copytree(Path({str(root / 'corpora')!r}), destination)
    else:
        real_command(*args, cwd=cwd)
''')
        for source in (fetch, backup_source, build, backup):
            execute(source)
        execute("assert EXPANSION_DATA_BACKED_UP is True\nassert all(item['verified'] for item in expansion_backup.values())")
        tokenizer_hash = sha256_file(project / "tokenizer/seed_v1/tokenizer.json")
        report = json.loads((project / "data/processed/expansion_v1/expansion_summary.json").read_text())
        assert report["tokenizer_sha256"] == tokenizer_hash
        inputs_before = _inventory(project / "data")
        # Published setup and every follow-on data cell can be repeated read-only.
        for source in (setup, fetch, backup_source, build, backup):
            execute(source)
        assert _inventory(project / "data") == inputs_before
    assert _inventory(drive / "seeds/seed_v1") == seed_before
