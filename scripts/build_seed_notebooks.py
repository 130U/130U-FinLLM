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
        SEED_TRAIN_SUCCEEDED = False
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
        '''), markdown("运行下面单元格开始短段训练。已有本版本检查点时自动校验并恢复最新进度；数据或运行时不兼容会明确报错。已经达到本段训练预算时只读取结果，保留原检查点和性能记录。"), code('''
        SEED_TRAIN_SUCCEEDED = False
        from finllm.runtime import restore_seed_checkpoint
        RESUME_CHECKPOINT = restore_seed_checkpoint(runtime_config["output_dir"], runtime_config["backup_dir"])
        arguments = ["train", "--config", runtime_path]
        if RESUME_CHECKPOINT:
            arguments += ["--resume", RESUME_CHECKPOINT]
        finllm(*arguments)
        RUN_SUMMARY = Path(runtime_config["output_dir"]) / "run_summary.json"
        require(RUN_SUMMARY, Path(runtime_config["output_dir"]) / "latest.pt",
                Path(runtime_config["output_dir"]) / "latest.pt.json")
        print(RUN_SUMMARY.read_text(encoding="utf-8"))
        SEED_TRAIN_SUCCEEDED = True
        '''), code('if not globals().get("SEED_TRAIN_SUCCEEDED", False):\n'
                   '    raise RuntimeError("训练单元格尚未成功结束；请先处理上方错误。现有备份保持不变。")\n'
                   + SEED_BACKUP),
        markdown("将上方显示的 GPU 型号及 run_summary.json 中的峰值显存、tokens/second、tokens 和验证 loss 带回工作对话后决定下一段预算。SEntFiN 的立场审核和资产映射仍是 SFT/DPO 的前置材料；本次不声称金融预测能力。")]
    write("02a_seed_pilot.ipynb", cells)

    cells = start("02b｜保留试跑成果并续训至累计 300 步", "先完成 02a。本页必须恢复 seed_v1 的已有检查点，先验证第 61→62 步的断点续训，再训练至累计 300 次更新（正常更新约 4,915,200 tokens）。所有模型、数据、优化器、学习率调度和精度设置保持原样，只延长停止步数。原始 61 步试跑报告会在 Drive 独立归档；不重新初始化模型。")
    cells[2] = code('''
        MOUNT_DRIVE = True
        from google.colab import drive
        drive.mount("/content/drive")
        DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
        ''')
    cells[3] = code('''
        CONTINUATION_SUCCEEDED = False
        RESUME_PROBE_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        PILOT_UPDATES = 61
        PROBE_UPDATES = 62
        TARGET_UPDATES = 300
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 运行时；不能在 CPU 上延续 T4 检查点。")
        from finllm.continuation import prepare_seed_continuation, run_seed_segment, backup_seed_continuation
        continuation = prepare_seed_continuation(
            PROJECT, DRIVE_ROOT, SEED_VERSION, pilot_updates=PILOT_UPDATES,
            probe_updates=PROBE_UPDATES, target_updates=TARGET_UPDATES)
        print("已恢复更新步数：", continuation["restored_updates"])
        print("原始试跑报告已验证并归档：", continuation["pilot_archive"])
        print("GPU:", torch.cuda.get_device_name(0))
        ''')
    cells += [markdown("先执行一次最短续训验证。已经完成第 62 步或后续步骤时，本单元格只校验并读取现有进度，不倒退、不重跑。若 Colab 的 PyTorch/CUDA 等环境与原检查点不同，会明确停止；请保留报错，不要通过改精度、批次大小或重新初始化绕过检查。"), code('''
        RESUME_PROBE_SUCCEEDED = False
        CONTINUATION_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        probe_result = run_seed_segment(continuation, "probe", finllm)
        print(json.dumps(probe_result, indent=2, ensure_ascii=False))
        RESUME_PROBE_SUCCEEDED = True
        '''), markdown("续训到累计 300 步，包含原先的 61 步。原始 pilot_config.json 保持不变。每 10 次更新自动备份检查点；若中断，重新从本页第一格执行即可恢复已保存进度。各段报告中的 updates/tokens 是累计计数，elapsed_seconds/tokens_per_second 则只计该段实际新训练。"), code('''
        CONTINUATION_SUCCEEDED = False
        CONTINUATION_BACKED_UP = False
        if not globals().get("RESUME_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先成功完成上方断点续训验证。")
        continuation_result = run_seed_segment(continuation, "continue", finllm)
        print(json.dumps(continuation_result, indent=2, ensure_ascii=False))
        CONTINUATION_SUCCEEDED = True
        '''), code('''
        CONTINUATION_BACKED_UP = False
        if not globals().get("CONTINUATION_SUCCEEDED", False):
            raise RuntimeError("续训尚未成功结束；保留现有备份，先处理上方错误。")
        backup_result = backup_seed_continuation(continuation)
        print(json.dumps(backup_result, indent=2, ensure_ascii=False))
        CONTINUATION_BACKED_UP = True
        print("累计续训与校验备份完成：", continuation["seed_drive"])
        print("原始试跑报告：", continuation["pilot_archive"])
        '''), markdown("完成后保留本页的续训报告与 verified: true 备份输出，再比较验证 loss 和资源消耗来决定下一段语料及预算。本页不会自动进入领域训练、SFT 或 DPO；不要修改原始 01a/02a 配置或删除 seed_v1 备份。")]
    write("02b_seed_continue.ipynb", cells)


if __name__ == "__main__":
    build()
