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
import os, sys, json, shutil, subprocess
from pathlib import Path

REPOSITORY = "https://github.com/130U/130U-FinLLM.git"
REF = "main"  # 首次解析并显示提交；复现时替换成完整的 40 位提交 SHA。
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
        raise RuntimeError("REF 不能唯一解析；请填写完整提交 SHA。")
    REVISION = refs[0].split()[0]

if not PROJECT.exists():
    command("git", "clone", "--no-checkout", REPOSITORY, PROJECT)
else:
    remote = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=PROJECT, text=True).strip()
    if remote.rstrip("/").removesuffix(".git") != REPOSITORY.removesuffix(".git"):
        raise RuntimeError("/content/130U-FinLLM 已属于其他仓库，请选择新目录。")
command("git", "fetch", "--depth", "1", "origin", REVISION, cwd=PROJECT)
command("git", "checkout", "--detach", REVISION, cwd=PROJECT)
os.chdir(PROJECT)
print("本次固定代码提交：", REVISION)

# 保留 Colab 自带的 CUDA / PyTorch；此处仅安装项目的其他依赖。
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
        raise RuntimeError(f"tokenizers {installed} 与现有环境不兼容：{incompatible}")
    print("tokenizers 兼容性检查通过：", installed)

if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
    raise RuntimeError("需要 PyTorch >= 2.5；请更换较新的 Colab 运行环境。")
tokenizer_requirements = tokenizer_install_requirements()
print("本次 tokenizers 约束：", tokenizer_requirements)
command(sys.executable, "-m", "pip", "install", "-q", "numpy>=1.26,<3", *tokenizer_requirements, "scikit-learn>=1.5,<2")
command(sys.executable, "-m", "pip", "install", "-q", "-e", ".", "--no-deps")
check_tokenizer_compatibility(tokenizer_requirements)

def finllm(*args):
    command(sys.executable, "-u", "-m", "finllm", *args, cwd=PROJECT)

def require(*paths):
    missing = [str(p) for p in paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError("缺少阶段输入，请先准备或从 Drive 恢复：\\n" + "\\n".join(missing))

finllm("environment")
disk = shutil.disk_usage("/content")
print(f"/content 剩余磁盘：{disk.free / 2**30:.2f} GiB")
Path("runtime_revision.txt").write_text(REVISION + "\\n")
'''

DRIVE = '''
# 可选持久化：需要你在 Google 的授权界面完成登录。
MOUNT_DRIVE = False  # 需要备份 / 恢复时设为 True，再执行此单元格。
DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
if MOUNT_DRIVE:
    from google.colab import drive
    drive.mount("/content/drive")
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print("Drive 项目目录：", DRIVE_ROOT)
else:
    print("尚未挂载 Drive；/content 是临时磁盘。正式训练前请启用备份。")
'''

RESTORE = '''
# 新 Notebook 可能分配新的虚拟机；按需恢复前一阶段的文件。
RESTORE_FROM_DRIVE = False
LIVE_STAGES_TO_RESTORE = []  # 中断恢复时填写，例如 ["pretrain"]；会覆盖本地对应阶段。
if RESTORE_FROM_DRIVE:
    if not MOUNT_DRIVE:
        raise RuntimeError("请先把 MOUNT_DRIVE 设为 True 并完成授权。")
    for folder in ("data", "tokenizer", "runs"):
        source = DRIVE_ROOT / folder
        if source.exists():
            finllm("sync", "--source", source, "--destination", PROJECT / folder, "--working-copy")
    for stage in LIVE_STAGES_TO_RESTORE:
        if stage not in ("pretrain", "domain", "sft", "dpo"):
            raise ValueError("未知检查点阶段：" + str(stage))
        source = DRIVE_ROOT / "live_checkpoints" / stage
        if not source.exists():
            raise FileNotFoundError("缺少阶段自动备份：" + str(source))
        finllm("sync", "--source", source, "--destination", PROJECT / "runs" / stage, "--working-copy")
    print("恢复完成。自动检查点只用于匹配原配置、tokenizer 和数据的继续训练；请指定 RESUME_CHECKPOINT。")
'''

BACKUP = '''
if MOUNT_DRIVE:
    for folder in ("data", "tokenizer", "runs"):
        source = PROJECT / folder
        if source.exists():
            finllm("sync", "--source", source, "--destination", DRIVE_ROOT / folder)
    shutil.copy2(PROJECT / "runtime_revision.txt", DRIVE_ROOT / "runtime_revision.txt")
    print("备份完成。请保留校验清单，下一运行时按清单恢复。")
else:
    print("成果当前仅在临时运行时中；请挂载 Drive 后重新运行备份单元格。")
'''


def start(title, details):
    return [markdown(f"# {title}\n\n130U-FinLLM｜130U 金融大语言模型\n\n{details}\n\n每个 Notebook 都包含独立环境入口。Colab 菜单：运行时 → 更改运行时类型 → T4 GPU。点击执行代码前可查看源码；所有输出均为实际执行后产生，仓库不预填训练成绩。"),
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
    target.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build():
    cells = start("00｜环境与完整技术试跑", "本页使用项目自编的微型样本，从零跑通 tokenizer、预训练、领域训练、SFT、DPO、校准和推理。它验证工程链路，不验证金融准确率。不会下载预训练权重或大规模语料。")
    cells += [code('''
        from datetime import datetime, timezone
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        SMOKE_OUTPUT = "runs/smoke/" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        finllm("smoke", "--output", SMOKE_OUTPUT, "--device", DEVICE)
        print("本次技术试跑目录：", SMOKE_OUTPUT)
        '''), markdown("试跑结果和正式训练始终分目录。不要把 smoke tokenizer 或检查点作为正式模型初始化。下一页需要自己的获准使用数据及冻结的评测原文池。"), code(BACKUP)]
    write("00_setup_and_smoke.ipynb", cells)

    cells = start("01｜语料与分词器", "本页处理本地已取得并审核的数据。将文件从 Drive 复制到 /content 后再处理。不要直接加载整个大型语料库，也不要把未知授权的社交媒体内容加入训练。具体格式见 docs/DATA_GUIDE.md。先准备任务评测原文池并划分，再清洗和排除预训练污染；最后训练 tokenizer 与打包。")
    cells += [markdown("下列输入路径是本项目约定。请先填写每条记录的来源、许可、目标和分组信息。原始文件不会由本页自动获取。正式配置的权限与分区校验失败时，先修复材料；不要绕开检查。"),
              code('''
        TASK_INPUT = "data/raw/task.jsonl"
        require(TASK_INPUT)
        finllm("prepare-task", "--input", TASK_INPUT, "--output", "data/task")
        '''),
              markdown("准备获准使用的通用和金融训练文本，并用冻结评测原文池排除污染。tokenizer_train.jsonl 应为两类**训练分区**的代表性混合，不包含 development、calibration 或 test。下面的命令只处理已准备的文件。"),
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
        '''), markdown("领域混合按 token 数指定：20M 金融＋5M 通用。若源分片小于预算，混合器明确记录循环复用次数，不能把重复 token 报成独立新语料。请查看输出清单中的计数。"),
              code('''
        finllm("mix-corpora", "--general", "data/processed/general/manifest.json", "--finance", "data/processed/finance/manifest.json", "--general-tokens", "5000000", "--finance-tokens", "20000000", "--output", "data/processed/domain")
        '''), code(BACKUP)]
    write("01_prepare_data.ipynb", cells)

    stages = [
        ("02_pretrain.ipynb", "02｜通用预训练", "pretrain", ["tokenizer/english/tokenizer.json", "data/processed/general/manifest.json", "data/processed/general_validation/manifest.json"],
         "默认 PILOT=True，从随机权重开始完成约 1M-token（61 次更新）资源试跑。检查实际资源与吞吐后，设 PILOT=False，并用 runs/pretrain/latest.pt 恢复到正式 100M-token 总预算。试跑保留正式学习率调度，已处理 token 计入总量。不能用 smoke 权重初始化。"),
        ("03_domain_pretrain.ipynb", "03｜金融领域继续预训练", "domain", ["runs/pretrain/best.pt", "tokenizer/english/tokenizer.json", "data/processed/domain/manifest.json", "data/processed/finance_validation/manifest.json"],
         "延续正式通用预训练选定的 best.pt，预算为 25M processed tokens。领域混合器按 80% 金融、20% 通用生成新的 token 分片，新增主体约 50MB；通用回放使用已有语料内容，不计作新语料。"),
        ("04_sft.ipynb", "04｜全参数监督微调", "sft", ["runs/domain/best.pt", "tokenizer/english/tokenizer.json", "data/task/train.jsonl", "data/task/development.jsonl"],
         "任务为针对指定资产的 bearish / neutral / bullish。损失只监督标签位置，更新全部模型参数。检查人工审核与类别分布；不会将普通 positive 标签直接当作 bullish。"),
        ("05_dpo.ipynb", "05｜DPO 偏好训练", "dpo", ["runs/sft/best.pt", "tokenizer/english/tokenizer.json", "data/task/preferences_train.jsonl", "data/task/development.jsonl"],
         "从选定的 SFT 检查点建立策略和冻结参考模型，使用训练分区的已审核偏好对。chosen 与 rejected 各只有一个标签，且两者不同。DPO 的效果不保证优于 SFT；在开发集上比较后再决定最终模型。"),
    ]
    for filename, title, stage, required, details in stages:
        cells = start(title, details)
        selection = ('PILOT = True\nCONFIG = "configs/pretrain_pilot.json" if PILOT else "configs/pretrain.json"' if stage == "pretrain"
                     else f'CONFIG = "configs/{stage}.json"')
        cells += [code(selection + '\nRESUME_CHECKPOINT = None  # 中断恢复或试跑转正式时，填写 "runs/' + stage + '/latest.pt"。\nrequire(CONFIG, ' + ', '.join(repr(p) for p in required) + ')\nprint(Path(CONFIG).read_text(encoding="utf-8"))'),
                  code('''
            # 正式训练前持久化输入，避免运行时中断后丢失 tokenizer 和数据。
            if not MOUNT_DRIVE:
                raise RuntimeError("请先启用 MOUNT_DRIVE 并完成授权，再备份正式训练输入。")
            for folder in ("data", "tokenizer"):
                finllm("sync", "--source", PROJECT / folder, "--destination", DRIVE_ROOT / folder)
            shutil.copy2(PROJECT / "runtime_revision.txt", DRIVE_ROOT / "runtime_revision.txt")
            print("训练输入已备份；训练中会另外自动同步完整检查点。")
            '''),
                  code('''
            RUN_TRAINING = False  # 阅读配置、检查 GPU 与数据后设为 True。
            if not RUN_TRAINING:
                raise RuntimeError("尚未启动正式训练。确认当前阶段与预算后，把 RUN_TRAINING 设为 True。")
            if not torch.cuda.is_available():
                raise RuntimeError("正式 24M 训练需要 GPU；请切换到 T4 运行时。")
            if not MOUNT_DRIVE:
                raise RuntimeError("正式训练使用 Drive 自动检查点备份，请先完成挂载。")
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

    cells = start("06｜选择模型、校准、评测与批量预测", "先比较 SFT 与 DPO 的开发集结果，固定最终检查点，再分别使用温度拟合集与阈值选择集。最后才打开 test。分数是三类立场概率的方向期望，不是收益率或价格上涨概率。")
    cells += [code('''
        require("tokenizer/english/tokenizer.json", "data/task/development.jsonl")
        for stage in ("sft", "dpo"):
            checkpoint = f"runs/{stage}/best.pt"
            require(checkpoint)
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", "tokenizer/english", "--input", "data/task/development.jsonl", "--output", f"runs/evaluation/{stage}_development.json")
            print(stage, Path(f"runs/evaluation/{stage}_development.json").read_text())
        '''), markdown("下面的 FINAL_CHECKPOINT 必须根据上面的开发集结果选择。校准文件绑定模型和 tokenizer 的 SHA256，更换模型后必须重新校准。不要根据 test 结果回头选择 SFT 或 DPO。"),
              code('''
        FINAL_CHECKPOINT = "runs/sft/best.pt"  # 按开发集结果选择。
        require(FINAL_CHECKPOINT, "data/task/calibration_temperature.jsonl", "data/task/calibration_threshold.jsonl")
        finllm("calibrate", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--temperature-data", "data/task/calibration_temperature.jsonl", "--threshold-data", "data/task/calibration_threshold.jsonl", "--output", "runs/evaluation/calibration.json")
        '''), code('''
        require("data/task/test.jsonl", "runs/evaluation/calibration.json")
        finllm("evaluate", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--input", "data/task/test.jsonl", "--calibration", "runs/evaluation/calibration.json", "--output", "runs/evaluation/test.json")
        print(Path("runs/evaluation/test.json").read_text())
        '''), markdown("批量预测输入为 JSONL，每行含 text、target，可选 language='en' 与经核查的 target_aliases。需要 cashtag 或明确名称关联；普通英文单词 CAT 不自动当作股票。超长、明显不支持的语言或目标关联不足的输入返回空分数并要求复核。"),
              code('''
        PREDICTION_INPUT = "data/raw/predict.jsonl"
        require(PREDICTION_INPUT)
        finllm("predict", "--checkpoint", FINAL_CHECKPOINT, "--tokenizer", "tokenizer/english", "--input", PREDICTION_INPUT, "--calibration", "runs/evaluation/calibration.json", "--output", "runs/predictions.jsonl")
        print("已写出：runs/predictions.jsonl")
        '''), code(BACKUP)]
    write("06_evaluate_and_predict.ipynb", cells)


if __name__ == "__main__":
    build()
