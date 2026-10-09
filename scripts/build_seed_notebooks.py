"""Build the bounded real-data path added after the initial smoke run."""
from build_notebooks import markdown, code, start, write


SEED_BACKUP = '''
SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
for name, source in (("raw", Path("data/raw") / SEED_VERSION),
                     ("processed", Path("data/processed") / SEED_VERSION),
                     ("tokenizer", Path("tokenizer") / SEED_VERSION),
                     ("run", Path("runs") / (SEED_VERSION + "_pretrain"))):
    if source.exists():
        finllm("sync", "--source", PROJECT / source, "--destination", SEED_DRIVE / name)
shutil.copy2(PROJECT / "runtime_revision.txt", SEED_DRIVE / "runtime_revision.txt")
print("本版本备份完成：", SEED_DRIVE)
'''


def build():
    cells = start("01a｜首批真实语料与 8K 词表", "下载固定版本的限量 Wikipedia 和 SEntFiN 候选标题，保存带日期和哈希的原创 FOMC 纪要快照。先冻结全部候选任务原文，再自训词表和分片。此页不会把新闻情绪自动改成看多/看空标签，也不会启动模型训练。预计原始文本为几十 MB，下载库仍有额外缓存。")
    cells[0]["source"] = [line.replace("Colab 菜单：运行时 → 更改运行时类型 → T4 GPU。",
        "本页准备语料可使用 CPU；下一本 02a 训练前再切换 T4 GPU。") for line in cells[0]["source"]]
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
        print("Drive 项目目录：", DRIVE_ROOT)
        ''')
    cells[3] = markdown("本页默认使用 seed_v1。重复运行时先校验并复用完整成果；新运行时会恢复 Drive 中本版本的快照。缺失或被修改的成果会明确报错并保留，届时选择新版本号或恢复完整备份。原始材料仅保存在当前运行时及你自己的 Drive，不上传 GitHub。")
    cells += [code('''
        SEED_VERSION = "seed_v1"
        ROOT = Path("data/raw") / SEED_VERSION
        CORPORA = ROOT / "corpora"
        TASK = ROOT / "task"
        PREPARED = ROOT / "prepared"
        TOKENIZER = Path("tokenizer") / SEED_VERSION
        PROCESSED = Path("data/processed") / SEED_VERSION
        RUN_OUTPUT = Path("runs") / (SEED_VERSION + "_pretrain")
        command(sys.executable, "-m", "pip", "install", "-q", "-e", ".[data]", cwd=PROJECT)
        SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
        for name, destination in (("raw", ROOT), ("processed", PROCESSED), ("tokenizer", TOKENIZER)):
            source = SEED_DRIVE / name
            if not destination.exists() and source.exists():
                require(source / "SNAPSHOT_COMPLETE.json")
                finllm("sync", "--source", source, "--destination", PROJECT / destination, "--working-copy")
        SEED_BUILD_ARGS = ["--corpora", CORPORA, "--task-pool", TASK / "task_exclusions.jsonl",
                           "--prepared", PREPARED, "--tokenizer", TOKENIZER,
                           "--processed", PROCESSED, "--run-output", RUN_OUTPUT]
        print("本页只准备数据；当前 CUDA 可用：", torch.cuda.is_available())
        '''), code('''
        command(sys.executable, "-u", "scripts/fetch_seed_corpora.py", "--output", CORPORA,
                "--general-max-bytes", "20000000", cwd=PROJECT)
        command(sys.executable, "-u", "scripts/prepare_sentfin_review.py", "--download", "--output", TASK, cwd=PROJECT)
        '''), code('''
        command(sys.executable, "-u", "scripts/build_seed_artifacts.py", *SEED_BUILD_ARGS, cwd=PROJECT)
        print((PROCESSED / "seed_summary.json").read_text(encoding="utf-8"))
        '''), code('# 备份前重新检查完整产物，不能把中断执行的半成品显示为成功。\n'
                   'command(sys.executable, "-u", "scripts/build_seed_artifacts.py", *SEED_BUILD_ARGS, '
                   '"--validate-only", cwd=PROJECT)\n' + SEED_BACKUP),
        markdown("完成后先查看 seed_summary.json 的真实 token 数。200 条任务审核表在 data/raw/seed_v1/task，填写的是你对目标资产立场的判断；原数据作者的人类情绪标签保留在单独列中。下一本 02a 可使用本页产物运行 24M 模型的约 1M-token 预训练试跑。")]
    write("01a_seed_data.ipynb", cells)

    cells = start("02a｜首批真实语料的 24M 预训练试跑", "从随机初始化开始，使用 seed_v1 的 8K tokenizer 和 Wikipedia 分片训练 61 次更新（约 1M tokens）。先完成 01a。本页只执行通用预训练的短段，不会自动进入 100M 长训练、SFT 或 DPO。")
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        ''')
    cells[3] = code('''
        SEED_VERSION = "seed_v1"
        SEED_DRIVE = DRIVE_ROOT / "seeds" / SEED_VERSION
        # 独立版本快照不会改写其他版本或小实验的备份清单。
        for name, relative, required in (
            ("processed", f"data/processed/{SEED_VERSION}", "seed_summary.json"),
            ("tokenizer", f"tokenizer/{SEED_VERSION}", "tokenizer.json")):
            destination = PROJECT / relative
            if not (destination / required).is_file():
                require(SEED_DRIVE / name / "SNAPSHOT_COMPLETE.json")
                finllm("sync", "--source", SEED_DRIVE / name,
                       "--destination", destination, "--working-copy")
        ''')
    cells += [code('''
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 运行时。")
        CONFIG = Path("data/processed") / SEED_VERSION / "pilot_config.json"
        require(CONFIG)
        runtime_config = json.loads(CONFIG.read_text(encoding="utf-8"))
        runtime_config["backup_dir"] = str(DRIVE_ROOT / "live_checkpoints" / (SEED_VERSION + "_pretrain"))
        runtime_path = Path("configs/runtime") / (SEED_VERSION + "_pilot.json")
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(runtime_config, indent=2), encoding="utf-8")
        print(json.dumps(runtime_config, indent=2))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown("运行下面单元格开始短段训练。已有本版本检查点时自动校验并恢复最新进度；数据或运行时不兼容会明确报错，不覆盖进度重新开始。"), code('''
        from finllm.runtime import restore_seed_checkpoint
        RESUME_CHECKPOINT = restore_seed_checkpoint(runtime_config["output_dir"], runtime_config["backup_dir"])
        arguments = ["train", "--config", runtime_path]
        if RESUME_CHECKPOINT:
            arguments += ["--resume", RESUME_CHECKPOINT]
        finllm(*arguments)
        print((Path(runtime_config["output_dir"]) / "run_summary.json").read_text(encoding="utf-8"))
        '''), code(SEED_BACKUP), markdown("将上方显示的 GPU 型号及 run_summary.json 中的峰值显存、tokens/second、tokens 和验证 loss 带回工作对话后决定下一段预算。SEntFiN 的立场审核和资产映射仍是 SFT/DPO 的前置材料；本次不声称金融预测能力。")]
    write("02a_seed_pilot.ipynb", cells)


if __name__ == "__main__":
    build()
