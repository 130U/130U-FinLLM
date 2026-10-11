"""Generate the data expansion and explicit weights-transfer Colab notebooks."""
from build_notebooks import BOOTSTRAP, markdown, code, write


DATA_BOOTSTRAP = BOOTSTRAP.replace('"-q", "-e", ".",', '"-q", "-e", ".[data]",')
if DATA_BOOTSTRAP == BOOTSTRAP:
    raise RuntimeError("Shared bootstrap install transaction could not be located")

DRIVE = '''
MOUNT_DRIVE = True
if not MOUNT_DRIVE:
    raise RuntimeError("本阶段需要已有种子输入及版本化备份，请完成 Drive 挂载。")
from google.colab import drive
drive.mount("/content/drive")
DRIVE_ROOT = Path("/content/drive/MyDrive/130U-FinLLM")
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
print("Drive 项目目录：", DRIVE_ROOT)
'''


def build():
    cells = [markdown('''
        # 01b｜扩充英文通用与金融语料，复用原 8K 词表

        130U-FinLLM｜130U 金融大语言模型

        先保留已完成的 seed_v1 数据、词表和 300 步成果。本页创建独立的 expansion_v1：
        固定 Wikipedia 快照默认最多 200MB，FOMC 纪要扩至 2011–2025；排除原 seed 文档与冻结任务文本，
        按文档切分新增 train/dev，使用原 tokenizer 编码。实际独立篇数与 tokens 完成后报告。
        不生成看多／看空标签，不重新训练词表，不启动模型训练。原文只保存在本运行时及自己的 Drive。

        本页可用 CPU。每格按顺序执行；断开后先等重新连接，确认已停止再重新运行。
        同版本重复运行先验证并复用完整产物；参数或输入变化时保留旧版本，另选新版本号。
        下载和 Parquet 解码有额外缓存，不能把 200MB 输出上限当作全部内存／空间占用。
        来源与权利说明见 docs/research/expanded_corpus_sources_20261011.md。
        '''), code(DATA_BOOTSTRAP), code(DRIVE), code('''
        EXPANSION_FETCH_SUCCEEDED = False
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        GENERAL_MAX_BYTES = 200_000_000
        MAX_GENERAL_DOCUMENTS = 50_000
        FINANCE_YEARS = ",".join(str(year) for year in range(2011, 2026))
        FINANCE_MAX_BYTES = 12_000_000
        MAX_FINANCE_DOCUMENTS = 128
        from finllm.expansion_runtime import restore_expansion_materials, expansion_build_arguments, backup_expansion_data
        from finllm.expansion import build_expansion_artifacts
        expansion_paths = restore_expansion_materials(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION)
        print(json.dumps({key: str(value) for key, value in expansion_paths.items()}, indent=2, ensure_ascii=False))
        print("本页只准备数据；当前 CUDA 可用：", torch.cuda.is_available())
        '''), markdown('''
        获取限量候选。已完成的来源输出只读校验；下载中断时校验进度与页面缓存后继续。
        不修改原 seed_v1。达到字节／篇数上限就停止，不承诺取得 100M 通用或 25M 独立金融 tokens。
        '''), code('''
        EXPANSION_FETCH_SUCCEEDED = False
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        command(sys.executable, "-u", "scripts/fetch_expansion_corpora.py",
                "--output", expansion_paths["corpora"],
                "--general-max-bytes", GENERAL_MAX_BYTES,
                "--max-general-documents", MAX_GENERAL_DOCUMENTS,
                "--finance-years", FINANCE_YEARS,
                "--finance-max-bytes", FINANCE_MAX_BYTES,
                "--max-finance-documents", MAX_FINANCE_DOCUMENTS, cwd=PROJECT)
        print((expansion_paths["corpora"] / "manifest.json").read_text(encoding="utf-8"))
        EXPANSION_FETCH_SUCCEEDED = True
        '''), code('''
        EXPANSION_CORPORA_BACKED_UP = False
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_FETCH_SUCCEEDED", False):
            raise RuntimeError("来源获取尚未成功；先处理上方错误，保留已有输入。")
        from finllm.runtime import sync_snapshot
        source_backup = sync_snapshot(expansion_paths["corpora"], expansion_paths["drive"] / "corpora")
        print(json.dumps(source_backup, indent=2, ensure_ascii=False))
        EXPANSION_CORPORA_BACKED_UP = True
        print("候选来源已校验并备份；随后开始清洗与编码。")
        '''), markdown('''
        使用原 8K 词表；排除 seed 的训练与验证原文身份、文本重复／词法近重复，以及冻结任务池。
        新增验证集仅用于当前语言模型诊断；原 seed 验证分片保留供 02c 比较训练前后损失。
        这仍是无标签语言模型语料，不能据此报告资产立场准确率。
        '''), code('''
        EXPANSION_DATA_SUCCEEDED = False
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_CORPORA_BACKED_UP", False):
            raise RuntimeError("请先成功备份候选来源，再构建扩充产物。")
        expansion_data_result = build_expansion_artifacts(**expansion_build_arguments(expansion_paths))
        print(json.dumps(expansion_data_result, indent=2, ensure_ascii=False))
        print((expansion_paths["processed"] / "expansion_summary.json").read_text(encoding="utf-8"))
        EXPANSION_DATA_SUCCEEDED = True
        '''), code('''
        EXPANSION_DATA_BACKED_UP = False
        if not globals().get("EXPANSION_DATA_SUCCEEDED", False):
            raise RuntimeError("数据构建尚未成功；不得将半成品报告为完整备份。")
        expansion_backup = backup_expansion_data(expansion_paths)
        print(json.dumps(expansion_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", expansion_paths["drive"] / "runtime_revision.txt")
        EXPANSION_DATA_BACKED_UP = True
        print("扩充数据与校验备份完成：", expansion_paths["drive"])
        '''), markdown('''
        完成标准：expansion_summary.json 记录真实新增文档／tokens；corpora、prepared、processed 三部分均 `verified: true`。
        下一本 02c 改用 T4，显式承接已完成 seed 300 步的模型权重。不要重做 01a/02a/02b，也不要直接运行旧 03。
        若本次候选不足以支持无回放的 610 步训练，02c 会停止并要求准备更多语料；不能用重复文本冒充新增数据。
        ''')]
    write("01b_expand_data.ipynb", cells)

    cells = [markdown('''
        # 02c｜承接 seed 权重，在扩充通用语料上训练新阶段

        130U-FinLLM｜130U 金融大语言模型

        先完成 01b 并确认三部分备份校验成功。本页要求 seed_v1 至少 300 步的正式预训练检查点，
        冻结其身份并沿用原 8K tokenizer。因为训练语料变化，采用明确的权重承接：优化器、调度器状态、
        数据游标及阶段计数重新建立。原 seed 检查点不覆盖；这不是同一数据流的精确断点续跑。

        默认先验证新阶段 0→1 步，再至新阶段累计 610 步，名义新增约 10M token 位置；
        实际有效 tokens 由报告给出，母模型约 4.91M tokens 单独列出。学习率调度总跨度为 6104 步，
        本段达到 610 步不表示 100M 全预算或金融大语言模型已训练完成。

        选择「运行时 → 更改运行时类型 → T4 GPU」，按顺序运行。
        重复同版本使用严格恢复；中断先等连接恢复、确认训练已停止，再从第一格重新执行。
        不要同时启动第二个训练进程。更改数据、词表或运行语义时选择独立实验版本。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = False
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        SEED_VERSION = "seed_v1"
        EXPANSION_VERSION = "expansion_v1"
        MIN_PARENT_UPDATES = 300
        PROBE_UPDATES = 1
        TARGET_UPDATES = 610
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 运行时；正式扩充阶段不在 CPU 上启动。")
        from finllm.expansion_runtime import prepare_expansion_training, run_expansion_segment, retention_check, backup_expansion_training
        expansion = prepare_expansion_training(
            PROJECT, DRIVE_ROOT, EXPANSION_VERSION, SEED_VERSION,
            min_parent_updates=MIN_PARENT_UPDATES, probe_updates=PROBE_UPDATES,
            target_updates=TARGET_UPDATES)
        print("已恢复新阶段更新步数：", expansion["restored_updates"])
        print("已冻结母检查点身份：", json.dumps(expansion["parent"], indent=2, ensure_ascii=False))
        print("本次新阶段配置：", json.dumps(expansion["config"], indent=2, ensure_ascii=False))
        print("GPU:", torch.cuda.get_device_name(0))
        '''), markdown('''
        先测冻结母检查点在原 seed 通用验证分片上的限量 loss；训练后在完全相同的位置比较。
        同时保留扩充语料自己的验证损失。两者都是文本预测诊断，均不是金融立场评分。
        已有输入身份完全相同的测量时读回结果。
        '''), code('''
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = False
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        retention_before = retention_check(expansion, "before")
        print(json.dumps(retention_before, indent=2, ensure_ascii=False))
        EXPANSION_RETENTION_BEFORE_SUCCEEDED = True
        '''), code('''
        EXPANSION_PROBE_SUCCEEDED = False
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_RETENTION_BEFORE_SUCCEEDED", False):
            raise RuntimeError("请先成功完成母检查点的原验证集测量。")
        expansion_probe = run_expansion_segment(expansion, "probe", finllm)
        print(json.dumps(expansion_probe, indent=2, ensure_ascii=False))
        EXPANSION_PROBE_SUCCEEDED = True
        '''), markdown('''
        0→1 步验证成功后才训练到新阶段累计 610 步。已有更靠后的同版本检查点时只校验恢复进度，不回退。
        每 10 步保存并同步 Drive；各段报告的 updates/tokens 属于新阶段累计计数，耗时只属于本次调用。
        数据／tokenizer／配置／运行时不兼容会停止；不要取消 provenance 检查或改成随机初始化绕过错误。
        '''), code('''
        EXPANSION_TRAIN_SUCCEEDED = False
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_RETENTION_BEFORE_SUCCEEDED", False) or not globals().get("EXPANSION_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先完成原验证集测量及单步权重承接验证。")
        expansion_result = run_expansion_segment(expansion, "continue", finllm)
        print(json.dumps(expansion_result, indent=2, ensure_ascii=False))
        EXPANSION_TRAIN_SUCCEEDED = True
        '''), code('''
        EXPANSION_RETENTION_AFTER_SUCCEEDED = False
        EXPANSION_TRAIN_BACKED_UP = False
        if not globals().get("EXPANSION_TRAIN_SUCCEEDED", False):
            raise RuntimeError("新阶段训练尚未成功结束；保留已有备份。")
        retention_after = retention_check(expansion, "after")
        print(json.dumps(retention_after, indent=2, ensure_ascii=False))
        EXPANSION_RETENTION_AFTER_SUCCEEDED = True
        '''), code('''
        EXPANSION_TRAIN_BACKED_UP = False
        if not all(globals().get(flag, False) for flag in (
            "EXPANSION_RETENTION_BEFORE_SUCCEEDED", "EXPANSION_PROBE_SUCCEEDED",
            "EXPANSION_TRAIN_SUCCEEDED", "EXPANSION_RETENTION_AFTER_SUCCEEDED")):
            raise RuntimeError("测量／单步验证／训练尚未全部成功；不得显示完整阶段备份。")
        expansion_train_backup = backup_expansion_training(expansion)
        print(json.dumps(expansion_train_backup, indent=2, ensure_ascii=False))
        shutil.copy2(PROJECT / "runtime_revision.txt", expansion["paths"]["drive"] / "runtime_revision.txt")
        EXPANSION_TRAIN_BACKED_UP = True
        print("扩充通用训练与校验备份完成：", expansion["paths"]["drive"])
        print("原 seed 数据与 300 步成果保留：", expansion["paths"]["seed_drive"])
        '''), markdown('''
        验收时保留单步与正式段报告、retention before/after、实际 token 数和 `verified: true` 最终备份输出。
        新阶段模型参数和词表不变；原 seed 300 步与此页新阶段步数分别记录。
        本页只训练扩充通用文本，不自动进入金融领域训练、SFT 或 DPO。
        金融扩充分片已由 01b 准备，下一阶段还需匹配版本的混合配置和入口；人工资产／立场标签仍待确认。
        ''')]
    write("02c_expanded_pretrain.ipynb", cells)


if __name__ == "__main__":
    build()
