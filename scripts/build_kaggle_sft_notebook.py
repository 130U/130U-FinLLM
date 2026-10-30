"""Generate the Kaggle training frontend for the user's trusted 74-step model."""
from build_notebooks import BOOTSTRAP, code, markdown, write


def build():
    bootstrap = BOOTSTRAP.replace('/content', '/kaggle/working')
    cells = [markdown('''
        # 04c｜Kaggle 单卡实体立场 SFT 试验

        使用 04b 导出的**本人私有**转运包及其 SHA256，承接 domain_v1 的 74 步权重。
        首次跨平台仅导入模型权重并建立新的优化器；不是恢复 Colab 的优化器。
        在 Kaggle 的 Add Input 添加**私有**数据集，包含 `130u_kaggle_transfer.zip`。
        Session options 选择 GPU T4 x2 和 Internet on。本试验只使用第一块 GPU。

        依次运行，记录第一格实际代码提交 SHA。以后跨会话恢复时若 main 已更新，
        请将第一格的 `REF = "main"` 改为首次运行时的完整 40 位代码 SHA，再依次运行。
        恢复只支持与原训练一致的代码、数据、路径、PyTorch/CUDA、设备和AMP配置。
        训练结果及暂存文件在
        `/kaggle/working`；它不是 Google Drive，运行时结束后不能保证保留。
        按输出提示下载恢复 ZIP 或保存私有 Notebook 版本与输出后，再结束会话。
        不要公开包含模型或原始标题的上传数据和输出。

        模型任务是标题内实体的看多／中性／看空研究试验，资产身份、语义事件
        和时序隔离尚未完成，开发集指标不代表证券预测或真实 X 效果。
        '''), code(bootstrap), code('''
        from finllm.kaggle_sft_runtime import (install_transfer_archive, install_transfer_directory,
            install_recovery_archive, install_recovery_directory)
        from finllm.sft_pilot_runtime import prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
        from pathlib import Path

        # 分别复制 04b 输出的转运清单 SHA256 与转运 ZIP SHA256。
        TRANSFER_MANIFEST_SHA256 = "请粘贴04b输出的转运清单SHA256".strip().lower()
        TRANSFER_ZIP_SHA256 = "请粘贴04b输出的转运ZIP的SHA256".strip().lower()
        # 跨会话恢复时，把之前的恢复 ZIP 作为私有 Input 挂载并填写其 SHA。
        # 首次运行保持空字符串；同一会话从本页重跑也不需要填写。
        RECOVERY_MANIFEST_SHA256 = "".strip().lower()
        RECOVERY_ZIP_SHA256 = "".strip().lower()
        INPUT_ROOT = Path("/kaggle/input")
        HANDOFF = Path("/kaggle/working/130u_handoff")
        manifests = sorted(INPUT_ROOT.rglob("transfer_manifest.json"))
        archives = sorted(INPUT_ROOT.rglob("130u_kaggle_transfer.zip"))
        if len(manifests) == 1:
            transfer = install_transfer_directory(manifests[0].parent, HANDOFF,
                                                  expected_manifest_sha256=TRANSFER_MANIFEST_SHA256)
        elif not manifests and len(archives) == 1:
            transfer = install_transfer_archive(archives[0], HANDOFF, expected_zip_sha256=TRANSFER_ZIP_SHA256)
        else:
            raise RuntimeError("请通过 Add Input 挂载唯一的私有转运数据集。")
        print("原成果校验完成：", json.dumps(transfer, indent=2, ensure_ascii=False))
        if RECOVERY_MANIFEST_SHA256 or RECOVERY_ZIP_SHA256:
            manifests = sorted(INPUT_ROOT.rglob("recovery_manifest.json"))
            candidates = sorted(INPUT_ROOT.rglob("130u_sft_recovery_*.zip"))
            if len(manifests) == 1:
                restored = install_recovery_directory(manifests[0].parent, HANDOFF,
                                         expected_manifest_sha256=RECOVERY_MANIFEST_SHA256)
            elif not manifests and len(candidates) == 1:
                restored = install_recovery_archive(candidates[0], HANDOFF,
                                         expected_zip_sha256=RECOVERY_ZIP_SHA256)
            else:
                raise RuntimeError("请挂载唯一的私有恢复数据集。")
            print("已恢复私有SFT进度：", json.dumps(restored, indent=2, ensure_ascii=False))
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("请在 Session options 选择 GPU T4，并从第一格重新运行。")
        print("GPU数量：", torch.cuda.device_count(), "本次使用：", torch.cuda.get_device_name(0))
        pilot = prepare_sft_pilot(PROJECT, HANDOFF, run_version="sft_kaggle_v1")
        print("已核验领域母模型和实际排除池：", json.dumps(pilot["parent_audit"], indent=2, ensure_ascii=False))
        print("任务分区：", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        '''), code('''
        from finllm.kaggle_sft_runtime import export_recovery
        from datetime import datetime, timezone
        SFT_PROBE_SUCCEEDED = False
        SFT_TRAIN_SUCCEEDED = False
        SFT_EVAL_SUCCEEDED = False
        probe = run_sft_pilot(pilot, "probe", finllm)
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        SFT_PROBE_SUCCEEDED = True
        recovery_file = Path("/kaggle/working") / ("130u_sft_recovery_probe_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
        print("单步恢复包（须下载或保存版本才持久）：", json.dumps(export_recovery(HANDOFF, recovery_file), indent=2, ensure_ascii=False))
        '''), code('''
        SFT_TRAIN_SUCCEEDED = False
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先完成单步检查。")
        result = run_sft_pilot(pilot, "continue", finllm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        SFT_TRAIN_SUCCEEDED = True
        '''), code('''
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False):
            raise RuntimeError("请先完成训练。")
        output = Path(pilot["config"]["output_dir"])
        development = pilot["task_local"] / "development.jsonl"
        for name, checkpoint in (("domain_before", Path(pilot["config"]["init_checkpoint"])),
                                 ("sft_best", output / "best.pt")):
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", pilot["config"]["tokenizer"],
                   "--input", development, "--output", output / (name + "_development.json"),
                   "--target-mode", "literal_entity")
            print(name, (output / (name + "_development.json")).read_text(encoding="utf-8"))
        SFT_EVAL_SUCCEEDED = True
        '''), code('''
        if not globals().get("SFT_TRAIN_SUCCEEDED", False) or not globals().get("SFT_EVAL_SUCCEEDED", False):
            raise RuntimeError("请先完成训练与开发集比较。")
        backup = backup_sft_pilot(pilot)
        print("运行结果本地备份：", json.dumps(backup, indent=2, ensure_ascii=False))
        recovery_file = Path("/kaggle/working") / ("130u_sft_recovery_final_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
        print("最终恢复包（请下载或保存私有版本输出）：", json.dumps(export_recovery(HANDOFF, recovery_file), indent=2, ensure_ascii=False))
        '''), markdown('''
        恢复包与转运包均含私有数据，只能保存在本人账户。保留恢复包 SHA、
        Notebook 首格代码 SHA、开发集结果与备份输出。Kaggle 进程或会话被突然回收时，
        未下载或保存的本地进度可能丢失；原领域权重仍在原 Drive。
        ''')]
    write("04c_kaggle_sft.ipynb", cells)


if __name__ == "__main__":
    build()
