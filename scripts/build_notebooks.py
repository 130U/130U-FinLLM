"""Generate the small, reviewable Colab front ends; no notebook executes on build."""
from pathlib import Path
import json
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": textwrap.dedent(text).strip().splitlines(keepends=True)}


BOOTSTRAP = '''
import os, sys, json, shutil, subprocess, importlib
from pathlib import Path

REPOSITORY = "https://github.com/130U/130U-FinLLM.git"
REF = "main"  # Resolve and display the commit first; use the full 40-character SHA for reproduction.
PROJECT = Path("/content/130U-FinLLM")

def command(*args, cwd=None):
    # Explicitly relay subprocess output into the notebook's captured stdout.
    with subprocess.Popen([str(x) for x in args], cwd=cwd, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)

if len(REF) == 40 and all(c in "0123456789abcdefABCDEF" for c in REF):
    REVISION = REF.lower()
else:
    refs = subprocess.check_output(["git", "ls-remote", REPOSITORY, REF], text=True).strip().splitlines()
    if len(refs) != 1:
        raise RuntimeError("REF does not resolve uniquely; enter the full commit SHA.")
    REVISION = refs[0].split()[0]

if not PROJECT.exists():
    command("git", "clone", "--no-checkout", REPOSITORY, PROJECT)
else:
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=PROJECT, text=True).strip()
    if remote.rstrip("/").removesuffix(".git") != REPOSITORY.removesuffix(".git"):
        raise RuntimeError("/content/130U-FinLLM belongs to another repository; choose a new directory.")
command("git", "fetch", "--depth", "1", "origin", REVISION, cwd=PROJECT)
command("git", "checkout", "--detach", REVISION, cwd=PROJECT)
os.chdir(PROJECT)
print("Pinned code revision:", REVISION)

# Keep the bundled Colab CUDA / PyTorch; install only the other project dependencies.
import torch
from importlib import metadata
from packaging.requirements import Requirement

def tokenizer_install_requirements():
    # Colab can include Transformers even though this project does not use it.
    # Pass its active tokenizer bounds to pip so an old runtime can retain 0.22,
    # while a current runtime (or an already broken one) can use 0.23.
    requirements = ["tokenizers>=0.21,<0.24"]
    try:
        existing = metadata.requires("transformers") or []
    except metadata.PackageNotFoundError:
        existing = []
    for value in existing:
        requirement = Requirement(value)
        if requirement.name.lower().replace("_", "-") == "tokenizers":
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                requirements.append(str(requirement))
    return list(dict.fromkeys(requirements))

def check_tokenizer_compatibility(requirements):
    installed = metadata.version("tokenizers")
    incompatible = [value for value in requirements
                    if installed not in Requirement(value).specifier]
    if incompatible:
        raise RuntimeError(f"tokenizers {installed} is incompatible with the existing environment:{incompatible}")
    print("tokenizers compatibility check passed:", installed)

def check_project_dependencies():
    # Read the refreshed installed metadata, not the previous editable checkout.
    problems = []
    for value in metadata.requires("finllm-130u") or []:
        requirement = Requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            problems.append(f"Missing {requirement.name}")
            continue
        if installed not in requirement.specifier:
            problems.append(f"{requirement.name}=={installed} does not satisfy {requirement}")
    if problems:
        raise RuntimeError("Project dependency check failed:" + "; ".join(problems))
    print("Installed project dependencies passed validation.")

def activate_project_in_kernel():
    # pip installs the editable .pth for NEW Python processes. A notebook kernel
    # that was already running does not process that .pth again automatically.
    source = (PROJECT / "src").resolve()
    expected = source / "finllm"
    if not (expected / "__init__.py").is_file():
        raise RuntimeError(f"Project Python source is missing:{expected}")
    sys.path[:] = [str(source)] + [entry for entry in sys.path if entry != str(source)]
    # Git may have changed code in the same directory since an earlier cell run.
    # Refresh only our modules; never reload Torch or its live CUDA state.
    for name in tuple(sys.modules):
        if name == "finllm" or name.startswith("finllm."):
            del sys.modules[name]
    importlib.invalidate_caches()
    package = importlib.import_module("finllm")
    runtime = importlib.import_module("finllm.runtime")
    for module, filename in ((package, "__init__.py"), (runtime, "runtime.py")):
        if Path(module.__file__).resolve() != expected / filename:
            raise RuntimeError(f"The notebook imported the wrong project location:{module.__file__}")
    if not callable(runtime.restore_seed_checkpoint):
        raise RuntimeError("The checkpoint restore interface is unavailable in this notebook.")
    print("Current notebook kernel import check passed:", runtime.__file__)

def check_fresh_process_import():
    # Verify the installed editable package separately, without injecting src
    # into PYTHONPATH. Both the current kernel and future CLI processes must work.
    probe = """import json, sys
from pathlib import Path
import finllm
import finllm.runtime as runtime
expected = Path(sys.argv[1]).resolve() / 'finllm'
for module, filename in ((finllm, '__init__.py'), (runtime, 'runtime.py')):
    if Path(module.__file__).resolve() != expected / filename:
        raise RuntimeError('Fresh process imported the wrong project: ' + str(module.__file__))
if not callable(runtime.restore_seed_checkpoint):
    raise RuntimeError('Fresh process checkpoint restore is unavailable')
print(json.dumps({'finllm': str(Path(finllm.__file__).resolve()), 'runtime': str(Path(runtime.__file__).resolve())}))
"""
    result = subprocess.check_output([sys.executable, "-c", probe, str(PROJECT / "src")],
                                     cwd=PROJECT, text=True)
    print("Fresh Python process import check passed:", result.strip())

if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
    raise RuntimeError("PyTorch >= 2.5 is required; select a newer Colab runtime.")
tokenizer_requirements = tokenizer_install_requirements()
print("Active tokenizers constraints:", tokenizer_requirements)
torch_version_before = metadata.version("torch")
torch_file_before = Path(torch.__file__).resolve()
# One resolver transaction replaces the previous finllm metadata AND its
# dependencies. Upgrading tokenizers first leaves old finllm bounds active and
# produces a misleading conflict warning even when the eventual install works.
# The exact installed Torch build is constrained, so pip keeps Colab's CUDA build.
command(sys.executable, "-m", "pip", "install", "-q", "-e", ".",
        "torch==" + torch_version_before, *tokenizer_requirements)
importlib.invalidate_caches()
if metadata.version("torch") != torch_version_before or Path(torch.__file__).resolve() != torch_file_before:
    raise RuntimeError("The PyTorch environment changed unexpectedly; preserve the output and stop training.")
check_tokenizer_compatibility(tokenizer_requirements)
check_project_dependencies()
activate_project_in_kernel()
check_fresh_process_import()
print("Notebook setup complete; PyTorch remains:", torch_version_before)

def finllm(*args):
    command(sys.executable, "-u", "-m", "finllm", *args, cwd=PROJECT)

def require(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("Stage inputs are missing; prepare them or restore them from Drive first:\\n" + "\\n".join(missing))

finllm("environment")
disk = shutil.disk_usage("/content")
print(f"Free disk space in /content:{disk.free / 2**30:.2f} GiB")
Path("runtime_revision.txt").write_text(REVISION + "\\n")
'''

DRIVE = '''
# Optional persistence: sign in through the Google authorization dialog.
MOUNT_DRIVE = False  # Set to True for backup / restore, then run this cell.
DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print("Drive project directory:", DRIVE_ROOT)
else:
    print("Drive is not mounted; /content is temporary storage. Enable backups before training.")
'''

RESTORE = '''
# A new notebook may receive a new VM; restore previous-stage files as needed.
RESTORE_FROM_DRIVE = False
LIVE_STAGES_TO_RESTORE = []  # Set for interrupted-run recovery, e.g. ["pretrain"]; overwrites the corresponding local stage.
if RESTORE_FROM_DRIVE:
    if not MOUNT_DRIVE:
        raise RuntimeError("Set MOUNT_DRIVE to True and complete authorization first.")
    for folder in ("data", "tokenizer", "runs"):
        source = DRIVE_ROOT / folder
        if source.exists():
            finllm("sync", "--source", source, "--destination", PROJECT / folder, "--working-copy")
    for stage in LIVE_STAGES_TO_RESTORE:
        if stage not in ("pretrain", "domain", "sft", "dpo"):
            raise ValueError("Unknown checkpoint stage:" + str(stage))
        source = DRIVE_ROOT / "live_checkpoints" / stage
        if not source.exists():
            raise FileNotFoundError("Stage automatic backup is missing:" + str(source))
        finllm("sync", "--source", source, "--destination", PROJECT / "runs" / stage, "--working-copy")
    print("Restore complete. Automatic checkpoints require matching configuration, tokenizer and data; set RESUME_CHECKPOINT to continue.")
'''

BACKUP = '''
if MOUNT_DRIVE:
    for folder in ("data", "tokenizer", "runs"):
        source = PROJECT / folder
        if source.exists():
            finllm("sync", "--source", source, "--destination", DRIVE_ROOT / folder)
    shutil.copy2(PROJECT / "runtime_revision.txt", DRIVE_ROOT / "runtime_revision.txt")
    print("Backup complete. Keep the verification manifest for restoration in the next runtime.")
else:
    print("Artifacts are only in temporary runtime storage; mount Drive and rerun the backup cell.")
'''


def start(title, details):
    return [markdown(f"# {title}\n\n130U-FinLLM | 130U Financial Language Model\n\n{details}\n\nEach notebook includes its own environment setup. Colab menu: Runtime > Change runtime type > T4 GPU. Inspect the source before execution; outputs are produced by real runs and the repository contains no prefilled training scores."),
            code(BOOTSTRAP), code(DRIVE), code(RESTORE)]


def write(name, cells):
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                             "language_info": {"name": "python"},
                             "colab": {"name": name, "provenance": []}},
                "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"cell-{index:03d}"
    target = ROOT / "notebooks" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def build():
    cells = start("00 | Environment and end-to-end smoke test", "This notebook uses project-authored tiny samples to exercise tokenizer training, pretraining, domain adaptation, SFT, DPO, calibration and inference from scratch. It checks the engineering pipeline, not financial accuracy. It downloads neither pretrained weights nor large corpora.")
    cells += [code('''
        from datetime import datetime, timezone
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        SMOKE_OUTPUT = "runs/smoke/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        finllm("smoke", "--output", SMOKE_OUTPUT, "--device", DEVICE)
        print("Smoke-test output directory:", SMOKE_OUTPUT)
        '''), markdown("Keep smoke-test and real-training outputs in separate directories. Do not initialize the real model from smoke tokenizers or checkpoints. The next notebook requires authorized data and a frozen pool of evaluation source texts."), code(BACKUP)]
    write("00_setup_and_smoke.ipynb", cells)

    cells = start("01 | Corpora and tokenizer", "This notebook processes data already acquired and reviewed locally. Copy files from Drive to /content before processing. Do not load an entire large corpus or train on social media content with unknown permissions. See docs/DATA_GUIDE.md for formats. Prepare and split the evaluation source-text pool first, then clean the corpora and remove pretraining contamination before training the tokenizer and packing tokens.")
    cells += [markdown("The input paths below follow project conventions. Complete each record's source, license, target and grouping fields first. This notebook does not acquire source files automatically. Resolve permission or partition validation failures in the inputs rather than bypassing the checks."),
              code('''
        TASK_INPUT = "data/raw/task.jsonl"
        require(TASK_INPUT)
        finllm("prepare-task", "--input", TASK_INPUT, "--output", "data/task")
        '''),
              markdown("Prepare authorized general and financial training texts and exclude the frozen evaluation source-text pool. tokenizer_train.jsonl should represent both **training partitions**, excluding development, calibration and test. The commands below process prepared files only."),
              code('''
        TOKENIZER_INPUT = "data/raw/tokenizer_train.jsonl"
        GENERAL_INPUT = "data/raw/general_train.jsonl"
        FINANCE_INPUT = "data/raw/finance_train.jsonl"
        GENERAL_VALIDATION = "data/raw/general_validation.jsonl"
        FINANCE_VALIDATION = "data/raw/finance_validation.jsonl"
        FROZEN_EVAL = "data/task/frozen_eval.jsonl"
        FROZEN_EXCLUSIONS = "data/task/frozen_exclusions.jsonl"
        require(TOKENIZER_INPUT, GENERAL_INPUT, FINANCE_INPUT, GENERAL_VALIDATION, FINANCE_VALIDATION, FROZEN_EVAL)
        finllm("freeze-exclusions", "--input", FROZEN_EVAL, "--input", GENERAL_VALIDATION, "--input", FINANCE_VALIDATION, "--output", FROZEN_EXCLUSIONS)
        finllm("train-tokenizer", "--input", TOKENIZER_INPUT, "--output", "tokenizer/english", "--vocab-size", "8192", "--exclude", FROZEN_EXCLUSIONS)
        finllm("pack", "--input", GENERAL_INPUT, "--tokenizer", "tokenizer/english", "--output", "data/processed/general", "--exclude", FROZEN_EXCLUSIONS)
        finllm("pack", "--input", FINANCE_INPUT, "--tokenizer", "tokenizer/english", "--output", "data/processed/finance", "--exclude", FROZEN_EXCLUSIONS)
        finllm("pack", "--input", GENERAL_VALIDATION, "--tokenizer", "tokenizer/english", "--output", "data/processed/general_validation", "--exclude", FROZEN_EVAL, "--partition", "validation")
        finllm("pack", "--input", FINANCE_VALIDATION, "--tokenizer", "tokenizer/english", "--output", "data/processed/finance_validation", "--exclude", FROZEN_EVAL, "--partition", "validation")
        '''), markdown("The domain mixture uses token budgets: 20M financial plus 5M general. If source shards are smaller than the budget, the mixer records replay counts explicitly. Repeated tokens are not independent new corpus data; inspect the output manifest counts."),
              code('''
        finllm("mix-corpora", "--general", "data/processed/general/manifest.json", "--finance", "data/processed/finance/manifest.json", "--general-tokens", "5000000", "--finance-tokens", "20000000", "--output", "data/processed/domain")
        '''), code(BACKUP)]
    write("01_prepare_data.ipynb", cells)

    stages = [
        ("02_pretrain.ipynb", "02 | General pretraining", "pretrain", ["tokenizer/english/tokenizer.json", "data/processed/general/manifest.json", "data/processed/general_validation/manifest.json"],
         "With PILOT=True, start from random weights for a resource pilot of about 1M tokens (61 updates). After reviewing resource use and throughput, set PILOT=False and resume from runs/pretrain/latest.pt toward the 100M-token total budget. The pilot retains the full learning-rate schedule and its processed tokens count toward that total. Do not initialize from smoke weights."),
        ("03_domain_pretrain.ipynb", "03 | Financial domain continued pretraining", "domain", ["runs/pretrain/best.pt", "tokenizer/english/tokenizer.json", "data/processed/domain/manifest.json", "data/processed/finance_validation/manifest.json"],
         "Continue from the selected general-pretraining best.pt with a budget of 25M processed tokens. The domain mixer generates new shards containing 80% financial and 20% general tokens, about 50 MB of token storage. General replay reuses existing content and is not new corpus data."),
        ("04_sft.ipynb", "04 | Full-parameter supervised fine-tuning", "sft", ["runs/domain/best.pt", "tokenizer/english/tokenizer.json", "data/task/train.jsonl", "data/task/development.jsonl"],
         "Predict bearish / neutral / bullish for the specified asset. Supervise only label positions while updating all model parameters. Check human review and class balance; generic positive sentiment is not automatically bullish stance."),
        ("05_dpo.ipynb", "05 | DPO preference training", "dpo", ["runs/sft/best.pt", "tokenizer/english/tokenizer.json", "data/task/preferences_train.jsonl", "data/task/development.jsonl"],
         "Initialize the policy and frozen reference model from the selected SFT checkpoint, using reviewed preference pairs from the training partition. chosen and rejected each contain one distinct label. DPO need not outperform SFT; compare development results before selecting the final model."),
    ]
    for filename, title, stage, required, details in stages:
        cells = start(title, details)
        selection = ('PILOT = True\nCONFIG = "configs/pretrain_pilot.json" if PILOT else "configs/pretrain.json"' if stage == "pretrain"
                     else f'CONFIG = "configs/{stage}.json"')
        cells += [code(selection + '\nRESUME_CHECKPOINT = None  # For recovery or extending the pilot, set "runs/' + stage + '/latest.pt".\nrequire(CONFIG, ' + ', '.join(repr(p) for p in required) + ')\nprint(Path(CONFIG).read_text(encoding="utf-8"))'),
                  code('''
            # Persist inputs before training so runtime interruptions do not lose the tokenizer or data.
            if not MOUNT_DRIVE:
                raise RuntimeError("Enable MOUNT_DRIVE and complete authorization before backing up training inputs.")
            for folder in ("data", "tokenizer"):
                finllm("sync", "--source", PROJECT / folder, "--destination", DRIVE_ROOT / folder)
            shutil.copy2(PROJECT / "runtime_revision.txt", DRIVE_ROOT / "runtime_revision.txt")
            print("Training inputs backed up; complete checkpoints will also sync automatically during training.")
            '''),
                  code('''
            RUN_TRAINING = False  # Set to True after reviewing the configuration, GPU and data.
            if not RUN_TRAINING:
                raise RuntimeError("Training has not started. Confirm the stage and budget, then set RUN_TRAINING to True.")
            if not torch.cuda.is_available():
                raise RuntimeError("24M training requires a GPU; switch to a T4 runtime.")
            if not MOUNT_DRIVE:
                raise RuntimeError("Training uses automatic Drive checkpoint backups; mount Drive first.")
            runtime_config = json.loads(Path(CONFIG).read_text(encoding="utf-8"))
            stage = runtime_config["stage"]
            runtime_config["backup_dir"] = str(DRIVE_ROOT / "live_checkpoints" / stage)
            runtime_path = Path("configs/runtime") / (stage + "_colab.json")
            runtime_path.parent.mkdir(parents=True, exist_ok=True)
            runtime_path.write_text(json.dumps(runtime_config, indent=2) + "\\n", encoding="utf-8")
            arguments = ["train", "--config", str(runtime_path)]
            if RESUME_CHECKPOINT:
                require(RESUME_CHECKPOINT, str(RESUME_CHECKPOINT) + ".json")
                arguments += ["--resume", RESUME_CHECKPOINT]
            finllm(*arguments)
            '''), code(BACKUP)]
        write(filename, cells)

    cells = start("06 | Model selection, calibration, evaluation and batch prediction", "Compare SFT and DPO development results, freeze the final checkpoint, then use separate temperature-fitting and threshold-selection sets. Open the test set last. The score is a directional expectation over three stance probabilities, not a return or probability of a price increase.")
    cells += [code('''
        require("tokenizer/english/tokenizer.json", "data/task/development.jsonl")
        for stage in ("sft", "dpo"):
            checkpoint = f"runs/{stage}/best.pt"
            require(checkpoint)
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", "tokenizer/english", "--input", "data/task/development.jsonl", "--output", f"runs/evaluation/{stage}_development.json")
            print(stage, Path(f"runs/evaluation/{stage}_development.json").read_text())
        '''), markdown("Select FINAL_CHECKPOINT below using the development results above. Calibration files bind the model and tokenizer SHA256 hashes; recalibrate after changing the model. Do not use test results to choose between SFT and DPO."),
              code('''
        FINAL_CHECKPOINT = "runs/sft/best.pt"  # Select using development results.
        require(FINAL_CHECKPOINT, "data/task/calibration_temperature.jsonl", "data/task/calibration_threshold.jsonl")
        finllm("calibrate", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--temperature-data", "data/task/calibration_temperature.jsonl", "--threshold-data", "data/task/calibration_threshold.jsonl", "--output", "runs/evaluation/calibration.json")
        '''), code('''
        require("data/task/test.jsonl", "runs/evaluation/calibration.json")
        finllm("evaluate", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--input", "data/task/test.jsonl", "--calibration", "runs/evaluation/calibration.json", "--output", "runs/evaluation/test.json")
        print(Path("runs/evaluation/test.json").read_text())
        '''), markdown("Batch input is JSONL with text and target on each line, optionally language='en' and verified target_aliases. A cashtag or explicit name association is required; the ordinary English word CAT is not automatically a stock. Overlong inputs, clearly unsupported languages or insufficient target association return a null score and require review."),
              code('''
        PREDICTION_INPUT = "data/raw/predict.jsonl"
        require(PREDICTION_INPUT)
        finllm("predict", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--input", PREDICTION_INPUT, "--calibration", "runs/evaluation/calibration.json", "--output", "runs/predictions.jsonl")
        print("Written: runs/predictions.jsonl")
        '''), code(BACKUP)]
    write("06_evaluate_and_predict.ipynb", cells)


if __name__ == "__main__":
    build()
