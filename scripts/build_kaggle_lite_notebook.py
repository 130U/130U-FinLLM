"""Generate the private Kaggle frontend for an existing lite weights handoff."""
from build_notebooks import BOOTSTRAP, code, markdown, write


def build():
    bootstrap = BOOTSTRAP.replace('/content', '/kaggle/working')
    cells = [markdown('''
        # 04e｜Kaggle 轻量承接已有金融模型并训练 SFT

        本页使用从 Colab 导出的私有轻量包，承接 `domain_v1` 已训练权重与原词表。
        不需要1.54 GiB的完整转运包。第一次只迁移模型权重，重新建立优化器；
        不是跨平台精确恢复旧优化器。原模型与标签包属于私人材料：请使用 Private 数据集和 Notebook。

        先在 Kaggle 添加轻量私有数据集；选择 GPU T4 x2 和 Internet on。
        实际只使用第一张卡。按顺序运行，并在配置格填写轻量包对应 SHA256。
        轻量包保留 Colab 导出时核查来源的报告，但 Kaggle 无法重新读取未转运的原始清单。
        训练结果是标题字面实体立场的探索性试验，非证券预测或 X 场景验证。
        '''), code(bootstrap), code('''
        from finllm.kaggle_lite import install_lite_archive, install_lite_directory, prepare_lite_sft
        from finllm.training import run_training
        from finllm.checkpoint import load_checkpoint, sha256_file
        from finllm.runtime import restore_seed_checkpoint
        from finllm.evaluation import run_evaluation
        import torch

        # 从轻量导出报告复制清单 SHA256；恢复时使用相同输入与代码版本。
        LITE_MANIFEST_SHA256 = "请填轻量包manifest_sha256".strip().lower()
        # 仅当 Kaggle 没有自动解压 ZIP 时，才需填写 ZIP 的 SHA256。
        LITE_ZIP_SHA256 = "".strip().lower()
        INPUT = Path("/kaggle/input")
        LITE = Path("/kaggle/working/130u_lite_input")
        RUN = Path("/kaggle/working/130u_lite_sft_v1")
        manifests = sorted(INPUT.rglob("lite_manifest.json"))
        archives = sorted(INPUT.rglob("130u_kaggle_lite.zip"))
        if len(manifests) == 1:
            installed = install_lite_directory(manifests[0].parent, LITE,
                                              expected_manifest_sha256=LITE_MANIFEST_SHA256)
        elif not manifests and len(archives) == 1:
            installed = install_lite_archive(archives[0], LITE, expected_zip_sha256=LITE_ZIP_SHA256)
        else:
            raise RuntimeError("请用 Add Input 添加唯一的 Private 轻量数据集。")
        print("轻量输入核验：", json.dumps(installed, indent=2, ensure_ascii=False))
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU，并从第一格重新运行。")
        print("GPU数量：", torch.cuda.device_count(), "本次使用：", torch.cuda.get_device_name(0))
        pilot = prepare_lite_sft(PROJECT, LITE, RUN,
                                 expected_manifest_sha256=installed["manifest_sha256"], epochs=2)
        config = pilot["config"]
        print("任务分区：", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        latest = restore_seed_checkpoint(RUN, Path(config["backup_dir"]))
        if latest:
            previous = load_checkpoint(latest, allow_previous=False)
            if previous.get("config", {}).get("lite_binding") != config.get("lite_binding"):
                raise ValueError("现有检查点与轻量输入身份不同；保留原实验并停止。")
            config.pop("init_checkpoint", None)
            config["resume"] = str(latest)
            print("检测到已有检查点，将按相同运行环境严格恢复。")
        result = run_training(config)
        print("SFT训练结果：", json.dumps(result, indent=2, ensure_ascii=False))
        '''), code('''
        for name, checkpoint in (("domain_before", LITE / "weights/domain_weights.pt"),
                                 ("sft_best", RUN / "best.pt")):
            metrics = run_evaluation(checkpoint, config["tokenizer"], config["validation_data"],
                                     RUN / (name + "_development.json"), target_mode="literal_entity")
            print(name, "开发集结果（探索性，不是最终测试）：", json.dumps(metrics, indent=2, ensure_ascii=False))
        print("模型及训练状态：", RUN)
        print("词表及迁移输入：", LITE)
        print("恢复检查点 SHA256：", sha256_file(RUN / "latest.pt"))
        print("恢复完整性标记 SHA256：", sha256_file(RUN / "latest.pt.json"))
        print("请保存 Private Notebook 版本和输出，或下载 latest.pt 及 latest.pt.json；")
        print("同时保留轻量 Private 数据集和第一格代码版本，工作目录不保证永久保存。")
        '''), markdown('''
        如果中途断线，同一会话保留文件时从第一格重跑。跨会话恢复须先保留并恢复原工作目录，
        固定第一格 `REF` 为首次运行的完整40位提交，且设备、PyTorch/CUDA、数据和路径一致。
        旧 Drive 权重不会被修改。开发集指标不是最终评估，原发布者内容权利仍需单独核验。
        ''')]
    write("04e_kaggle_lite_sft.ipynb", cells)


if __name__ == "__main__":
    build()
