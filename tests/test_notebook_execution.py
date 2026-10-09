"""Execute the Colab front end in real, long-lived Jupyter kernels.

Only external services are substituted: Git uses a local checkout, installed
dependencies are reused offline, and Drive is a temporary local directory.
Editable installation, kernel imports, project CLI, snapshots, model updates,
and checkpoint recovery are real. A tiny CPU fixture replaces the GPU budget;
passing this test is not evidence that a T4 training run has completed.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import sysconfig
import time
import venv

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
import numpy as np

from finllm.checkpoint import load_checkpoint, sha256_file
from finllm.runtime import sync_snapshot
from finllm.tokenizer import train_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def _run(*args, cwd=None, env=None):
    # Decoding stdout as UTF-8 does not set the CHILD's stdout encoding.
    # In particular Windows otherwise fails when printing a Chinese path.
    result = subprocess.run([str(value) for value in args], cwd=cwd,
                            env=_offline_environment() if env is None else env,
                            capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=120)
    if result.returncode:
        raise AssertionError(f"Fixture command failed ({result.returncode}): {result.args!r}\n"
                             f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return result


def _offline_environment():
    env = dict(os.environ)
    # pytest's source path and user site must not accidentally make finllm
    # importable before the notebook installs and activates it.
    env.pop("PYTHONPATH", None)
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PIP_NO_INDEX="1",
               PIP_DISABLE_PIP_VERSION_CHECK="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false",
               JUPYTER_PLATFORM_DIRS="1")
    return env


def _isolated_python(directory):
    # pip itself is reused with the other installed dependencies below.
    venv.EnvBuilder(with_pip=False).create(directory)
    python = directory / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    site = Path(_run(python, "-c", "import sysconfig; print(sysconfig.get_path('purelib'))").stdout.strip())
    # Reuse large installed dependencies, but do NOT process their .pth files.
    # A --system-site-packages venv would inherit the project's existing
    # editable .pth and conceal the very Colab failure being tested.
    (site / "offline_test_dependencies.pth").write_text(
        sysconfig.get_path("purelib") + "\n", encoding="utf-8")
    return python


def _local_repository(path):
    path.mkdir()
    shutil.copytree(ROOT / "src/finllm", path / "src/finllm",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for directory in ("configs", "scripts"):
        shutil.copytree(ROOT / directory, path / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "pyproject.toml", path / "pyproject.toml")
    env = _offline_environment()
    env.update(GIT_AUTHOR_NAME="Notebook integration fixture", GIT_AUTHOR_EMAIL="fixture@example.invalid",
               GIT_COMMITTER_NAME="Notebook integration fixture", GIT_COMMITTER_EMAIL="fixture@example.invalid")
    _run("git", "init", "-b", "main", path, env=env)
    _run("git", "add", ".", cwd=path, env=env)
    _run("git", "commit", "-qm", "Local notebook execution fixture", cwd=path, env=env)
    return path


@contextmanager
def _kernel(python, directory):
    specs = directory / "kernels"
    spec = specs / "finllm-integration"
    spec.mkdir(parents=True)
    (spec / "kernel.json").write_text(json.dumps({
        "argv": [str(python), "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "FinLLM isolated notebook test", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(kernel_name="finllm-integration", connection_file=str(directory / "connection.json"),
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(specs)]))
    env = _offline_environment()
    env.update(IPYTHONDIR=str(directory / "ipython"), JUPYTER_RUNTIME_DIR=str(directory / "jupyter"))
    manager.start_kernel(cwd=str(directory), env=env)
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)

        def execute(source, *, expected_error=None):
            identifier = client.execute(source, stop_on_error=True)
            output, error = [], None
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                try:
                    message = client.get_iopub_msg(timeout=1)
                except queue.Empty:
                    continue
                if message.get("parent_header", {}).get("msg_id") != identifier:
                    continue
                content = message["content"]
                if message["msg_type"] == "stream":
                    output.append(content["text"])
                elif message["msg_type"] == "error":
                    error = content
                elif message["msg_type"] == "status" and content["execution_state"] == "idle":
                    break
            else:
                raise AssertionError("Notebook cell exceeded 180 seconds\n" + "".join(output))
            if expected_error:
                assert error and error["ename"] == expected_error, (error, "".join(output))
                return error
            assert error is None, "\n".join(error["traceback"]) + "\n" + "".join(output) if error else ""
            return "".join(output)

        yield execute
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)


class _OfflineNotebook(ast.NodeTransformer):
    def __init__(self, repository, project):
        self.repository, self.project = repository, project

    def visit_Assign(self, node):
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name == "REPOSITORY":
                node.value = ast.Constant(str(self.repository))
            elif name == "PROJECT":
                node.value = ast.parse(f"Path({str(self.project)!r})", mode="eval").body
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "command":
            constants = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
            if "pip" in constants and "install" in constants:
                if "-e" not in constants:
                    return ast.parse("print('Offline test: dependencies already installed')", mode="eval").body
                node.args = [ast.parse("sys.executable", mode="eval").body] + [ast.Constant(x) for x in
                    ("-m", "pip", "install", "-q", "-e", ".", "--no-deps", "--no-build-isolation", "--ignore-installed")]
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "disk_usage"
                and node.args and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "/content"):
            node.args[0] = ast.Constant(str(self.project.parent))
        return self.generic_visit(node)


def _offline(source, repository, project):
    tree = _OfflineNotebook(repository, project).visit(ast.parse(source))
    return ast.unparse(ast.fix_missing_locations(tree))


def _cells(name):
    return ["".join(cell["source"]) for cell in
            json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))["cells"]
            if cell["cell_type"] == "code"]


def _cpu_config_cell(source):
    """Explicit test-only hardware substitution; never fake torch.cuda state."""
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.If) and "torch.cuda.is_available" in ast.unparse(node.test)
    ) and not (
        isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and "torch.cuda.get_device_name" in ast.unparse(node.value)
    )]
    return ast.unparse(tree)


def _drive_seed(root):
    source = root / "seed-fixture"
    tokenizer = source / "tokenizer"
    tok = train_tokenizer(["Original CPU integration fixture financial text " * 30,
                           "Testing restore checkpoints neutral bullish bearish " * 30], tokenizer, 384)
    processed = source / "processed"
    processed.mkdir()
    data = processed / "tokens.bin"
    (np.arange(257, dtype=np.uint16) % (tok.get_vocab_size() - 6) + 6).tofile(data)
    (processed / "manifest.json").write_text(json.dumps({
        "format": "finllm-uint16-v1", "mode": "smoke", "vocab_size": tok.get_vocab_size(),
        "tokenizer_sha256": sha256_file(tokenizer / "tokenizer.json"), "token_count": 257,
        "source_sha256": "project-authored-integration-fixture", "shards": [
            {"path": data.name, "count": 257, "sha256": sha256_file(data)}],
    }), encoding="utf-8")
    config = {
        "mode": "smoke", "stage": "pretrain", "tokenizer": "tokenizer/seed_v1",
        "data": "data/processed/seed_v1/manifest.json", "validation_data": "data/processed/seed_v1/manifest.json",
        "output_dir": "runs/seed_v1_pretrain",
        "model": {"vocab_size": tok.get_vocab_size(), "hidden_size": 16, "num_hidden_layers": 1,
                  "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 32,
                  "max_position_embeddings": 32},
        "train": {"device": "cpu", "amp": False, "batch_size": 2, "grad_accumulation": 1,
                  "seq_len": 16, "max_updates": 2, "schedule_updates": 3, "warmup_updates": 0,
                  "seed": 130, "save_every": 1, "learning_rate": 0.001, "validation_max_batches": 1},
    }
    (processed / "pilot_config.json").write_text(json.dumps(config), encoding="utf-8")
    (processed / "seed_summary.json").write_text(json.dumps({"synthetic_test_only": True}), encoding="utf-8")
    drive = root / "drive"
    for name in ("processed", "tokenizer"):
        sync_snapshot(source / name, drive / "seeds/seed_v1" / name)
    return drive


def test_live_kernel_install_and_seed_notebook_checkpoint_roundtrip(tmp_path, monkeypatch):
    # The harness must configure its subprocesses independently of the shell.
    monkeypatch.delenv("PYTHONUTF8", raising=False)
    print("Notebook integration: create isolated Python and local Git fixture", flush=True)
    python = _isolated_python(tmp_path / "python")
    repository = _local_repository(tmp_path / "upstream")
    project = tmp_path / "runtime" / "130U-FinLLM"
    project.parent.mkdir()
    drive = _drive_seed(tmp_path)
    cells = _cells("02a_seed_pilot.ipynb")
    bootstrap = _offline(cells[0], repository, project)
    restore = next(source for source in cells if 'for name, relative, required' in source)
    config = next(source for source in cells if 'runtime_config = json.loads' in source)
    train = next(source for source in cells if 'from finllm.runtime import restore_seed_checkpoint' in source)
    backup = next(source for source in cells if 'for name, source in' in source)
    prepare_inputs = next(source for source in _cells("01a_seed_data.ipynb") if "SEED_BUILD_ARGS =" in source)
    evidence = {"device": "cpu", "synthetic_test_only": True}
    first = tmp_path / "first-kernel"
    first.mkdir()
    print("Notebook integration: start live kernel before editable install", flush=True)
    with _kernel(python, first) as execute:
        execute("import importlib.util\nassert importlib.util.find_spec('finllm') is None")
        # The old notebook installed into an already-running kernel. A fresh
        # CLI sees editable .pth changes; that live kernel does not.
        execute(f"import subprocess, sys\nsubprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', {str(repository)!r}, '--no-deps', '--no-build-isolation', '--ignore-installed'], check=True)")
        output = execute("print(subprocess.check_output([sys.executable, '-m', 'finllm', 'environment'], text=True))")
        assert '"python"' in output
        error = execute("from finllm.runtime import restore_seed_checkpoint", expected_error="ModuleNotFoundError")
        assert "finllm" in error["evalue"]
        evidence["old_live_kernel_import_error"] = error["ename"]
        print("Notebook integration: reproduced old live-kernel import failure", flush=True)
        execute(bootstrap)
        execute("from finllm.runtime import restore_seed_checkpoint\nassert callable(restore_seed_checkpoint)")
        # Repeat execution in the SAME kernel is part of the user workflow.
        execute(bootstrap)
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(restore)
        execute(_offline(prepare_inputs, repository, project))
        execute(_cpu_config_cell(config))
        execute(backup, expected_error="RuntimeError")
        execute(train)
        first_summary = json.loads((project / "runs/seed_v1_pretrain/run_summary.json").read_text())
        assert first_summary["updates"] == 2
        execute(backup)
        before = load_checkpoint(project / "runs/seed_v1_pretrain/latest.pt")
        saved_artifacts = {path.name: (sha256_file(path), path.stat().st_mtime_ns)
                           for path in (project / "runs/seed_v1_pretrain").iterdir() if path.is_file()}
        execute(train)
        after = load_checkpoint(project / "runs/seed_v1_pretrain/latest.pt")
        assert before["progress"] == after["progress"]
        assert saved_artifacts == {path.name: (sha256_file(path), path.stat().st_mtime_ns)
                                   for path in (project / "runs/seed_v1_pretrain").iterdir() if path.is_file()}
        evidence["same_kernel_repeat_updates"] = after["progress"]["updates"]
        print("Notebook integration: bootstrap, training, backup and repeat passed", flush=True)

    # Recreate the ephemeral runtime at a DIFFERENT path; inputs and progress
    # must come back from the persistent Drive fixture without manual copying.
    restored_project = tmp_path / "new-runtime" / "130U-FinLLM"
    restored_project.parent.mkdir()
    fresh_python = _isolated_python(tmp_path / "fresh-python")
    second = tmp_path / "second-kernel"
    second.mkdir()
    print("Notebook integration: restore into a fresh Python, kernel and runtime directory", flush=True)
    with _kernel(fresh_python, second) as execute:
        execute(_offline(cells[0], repository, restored_project))
        execute(f"MOUNT_DRIVE = True\nDRIVE_ROOT = Path({str(drive)!r})")
        execute(_offline(prepare_inputs, repository, restored_project))
        execute(restore)
        execute(_cpu_config_cell(config))
        # Extend only the stopping budget, preserving the scheduler and inputs.
        execute("runtime_config['train']['max_updates'] = 3\nruntime_path.write_text(json.dumps(runtime_config), encoding='utf-8')")
        output = execute(train)
        execute(backup)
        result = json.loads((restored_project / "runs/seed_v1_pretrain/run_summary.json").read_text())
        assert result["updates"] == 3
        assert result["tokens"] > first_summary["tokens"]
        assert sha256_file(restored_project / "runs/seed_v1_pretrain/latest.pt") == sha256_file(
            drive / "live_checkpoints/seed_v1_pretrain/latest.pt")
        evidence.update(restored_new_kernel_updates=result["updates"],
                        restored_new_kernel_tokens=result["tokens"],
                        live_checkpoint_and_local_sha256_match=True)
    print("COLAB_KERNEL_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
