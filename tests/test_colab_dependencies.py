"""Regression for the Colab Transformers 5.18 / tokenizers 0.22 conflict."""

import ast
import importlib.util
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("colab_builder", ROOT / "scripts/build_notebooks.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def bootstrap_helpers(transformers_requirements, installed_version):
    def requires(name):
        assert name == "transformers"
        if transformers_requirements is None:
            raise metadata.PackageNotFoundError(name)
        return transformers_requirements

    # Execute the exact notebook helper definitions, without running Git or pip.
    nodes = [node for node in ast.parse(builder.BOOTSTRAP).body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"tokenizer_install_requirements", "check_tokenizer_compatibility"}]
    namespace = {
        "Requirement": Requirement,
        "metadata": SimpleNamespace(requires=requires,
                                    version=lambda name: installed_version,
                                    PackageNotFoundError=metadata.PackageNotFoundError),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "colab_helpers", "exec"), namespace)
    return namespace


def test_current_colab_conflict_is_repaired_by_joint_requirements():
    helpers = bootstrap_helpers(["tokenizers>=0.23.1,<0.24.0", "huggingface-hub>=1.0"], "0.23.1")
    requirements = helpers["tokenizer_install_requirements"]()
    assert all("0.23.1" in Requirement(value).specifier for value in requirements)
    assert not all("0.22.2" in Requirement(value).specifier for value in requirements)
    helpers["check_tokenizer_compatibility"](requirements)


def test_still_conflicting_install_is_not_reported_as_success():
    helpers = bootstrap_helpers(["tokenizers>=0.23.1,<0.24.0"], "0.22.2")
    with pytest.raises(RuntimeError, match="0.22.2"):
        helpers["check_tokenizer_compatibility"](helpers["tokenizer_install_requirements"]())


@pytest.mark.parametrize("transformers_requirements", [None, ["tokenizers>=0.22,<0.23"]])
def test_compatible_existing_older_runtime_does_not_need_tokenizer_upgrade(transformers_requirements):
    helpers = bootstrap_helpers(transformers_requirements, "0.22.2")
    requirements = helpers["tokenizer_install_requirements"]()
    assert all("0.22.2" in Requirement(value).specifier for value in requirements)
    helpers["check_tokenizer_compatibility"](requirements)


def test_inactive_optional_requirement_does_not_constrain_install():
    helpers = bootstrap_helpers(['tokenizers<0.22; extra == "legacy"',
                                 'tokenizers<0.22; python_version < "3.0"'], "0.23.1")
    requirements = helpers["tokenizer_install_requirements"]()
    assert len(requirements) == 1
    helpers["check_tokenizer_compatibility"](requirements)


def test_project_and_notebook_share_the_supported_tokenizer_range():
    # Python 3.10 remains supported, so do not import the 3.11-only tomllib here.
    project_dependencies = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    helpers = bootstrap_helpers(None, "0.23.1")
    requirement, = helpers["tokenizer_install_requirements"]()
    assert f'"{requirement}"' in project_dependencies
