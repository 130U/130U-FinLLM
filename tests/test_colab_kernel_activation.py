"""Notebook kernels need explicit activation after an editable install."""

import ast
import importlib.util
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("kernel_colab_builder", ROOT / "scripts/build_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def helper_source(name):
    node, = [node for node in ast.parse(builder.BOOTSTRAP).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    return ast.unparse(node)


def test_activation_refreshes_old_project_modules_without_reloading_torch(tmp_path):
    project = tmp_path / "new_project"
    source = project / "src" / "finllm"
    source.mkdir(parents=True)
    (source / "__init__.py").write_text("VERSION = 'current'\n", encoding="utf-8")
    (source / "runtime.py").write_text("def restore_seed_checkpoint(*args): return 'current'\n", encoding="utf-8")
    script = "\n".join([
        "import importlib, sys, types",
        "from pathlib import Path",
        "PROJECT = Path(sys.argv[1])",
        # Keep this interpreter independent of the pytest process's imported modules.
        "torch_sentinel = types.ModuleType('torch')",
        "sys.modules['torch'] = torch_sentinel",
        "unrelated = types.ModuleType('finllm_plugin')",
        "sys.modules['finllm_plugin'] = unrelated",
        "old = types.ModuleType('finllm')",
        "old.__file__ = '/old_checkout/src/finllm/__init__.py'",
        "sys.modules['finllm'] = old",
        "sys.modules['finllm.runtime'] = types.ModuleType('finllm.runtime')",
        helper_source("activate_project_in_kernel"),
        "activate_project_in_kernel()",
        "from finllm.runtime import restore_seed_checkpoint",
        "assert restore_seed_checkpoint() == 'current'",
        "assert sys.modules['finllm'] is not old",
        "assert sys.modules['torch'] is torch_sentinel",
        "assert sys.modules['finllm_plugin'] is unrelated",
        "activate_project_in_kernel()",
        "assert sys.path.count(str((PROJECT / 'src').resolve())) == 1",
        "assert sys.modules['torch'] is torch_sentinel",
    ])
    result = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", script, str(project)],
                            text=True, encoding="utf-8", capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("当前 Notebook 内核导入检查通过") == 2


def dependency_helper(requirements, versions):
    def version(name):
        if name not in versions:
            raise metadata.PackageNotFoundError(name)
        return versions[name]

    namespace = {"Requirement": Requirement,
                 "metadata": SimpleNamespace(requires=lambda name: requirements,
                                             version=version,
                                             PackageNotFoundError=metadata.PackageNotFoundError)}
    exec(helper_source("check_project_dependencies"), namespace)
    return namespace["check_project_dependencies"]


def test_stale_project_tokenizer_metadata_is_rejected_after_install():
    check = dependency_helper(["torch>=2.5,<3", "tokenizers>=0.21,<0.23"],
                              {"torch": "2.11.0+cu126", "tokenizers": "0.23.3"})
    with pytest.raises(RuntimeError, match="tokenizers==0.23.3"):
        check()


def test_current_project_metadata_and_inactive_extras_are_checked():
    check = dependency_helper(["torch>=2.5,<3", "tokenizers>=0.21,<0.24",
                               'pytest>=8; extra == "test"'],
                              {"torch": "2.11.0+cu126", "tokenizers": "0.23.3"})
    check()


def test_missing_base_dependency_fails_readiness_check():
    check = dependency_helper(["numpy>=1.26,<3"], {})
    with pytest.raises(RuntimeError, match="numpy"):
        check()
