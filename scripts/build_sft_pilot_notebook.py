"""Generate a Colab frontend for the reviewed literal-entity SFT pilot."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04a｜从金融领域模型开始实体立场 SFT 试验

        本页承接 Drive 中 `domain_v1` 的74步检查点和原8K词表，首次迁移权重并新建优化器。
        它训练人工确认的新闻标题对原文字面实体的立场，属于本地研究试验；目标不是已核验的证券代码。
        本数据未完成语义事件审查或历史时点核验，开发集指标不代表真实推特或未来市场准确率。
        原报道可能存在第三方权利，当前只用于本人的研究试验；请保留来源说明。

        选择T4，从第一格依次运行。首次需要上传本机 `entity_v1_upload.zip`；之后复用Drive备份。
        断线后确认旧进程已停止，从头运行恢复。记录首格显示的40位代码提交；若仓库main后来更新，
        将首格的REF改为首次使用的完整提交后恢复。不要同时启动两份训练。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.sft_pilot_runtime import install_task_zip, prepare_sft_pilot, run_sft_pilot, backup_sft_pilot
        from finllm.expansion_runtime import _inventory
        TASK_VERSION = "entity_v1"
        RUN_VERSION = "sft_entity_v1"
        TASK_ZIP_SHA256 = "e87d12013271eb8f93f926b0b2c71cef15b0c777cc58d6e08348773dbd6a7be6"
        TASK_DRIVE = DRIVE_ROOT / "tasks" / TASK_VERSION
        if not (TASK_DRIVE / "SNAPSHOT_COMPLETE.json").exists():
            from google.colab import files
            import tempfile
            with tempfile.TemporaryDirectory(prefix="finllm-upload-") as folder:
                original_cwd = Path.cwd()
                try:
                    os.chdir(folder)
                    print("请选择本机 entity_v1_upload.zip，上传后会校验并备份到 Drive。")
                    uploaded = files.upload()
                    if len(uploaded) != 1:
                        raise ValueError("请选择且只选择一个任务 ZIP。")
                    name = next(iter(uploaded))
                    if Path(name).name != name:
                        raise ValueError("上传文件名无效。")
                    result = install_task_zip(Path(folder) / name, DRIVE_ROOT, TASK_ZIP_SHA256, TASK_VERSION)
                finally:
                    os.chdir(original_cwd)
            print("任务数据备份：", json.dumps(result, indent=2, ensure_ascii=False))
        else:
            _inventory(TASK_DRIVE)
            print("已复用并校验任务数据快照：", TASK_DRIVE)
        '''), code('''
        if not torch.cuda.is_available():
            raise RuntimeError("请选择 T4 GPU 后从第一格重新运行。")
        pilot = prepare_sft_pilot(PROJECT, DRIVE_ROOT, task_version=TASK_VERSION, run_version=RUN_VERSION)
        print("已核验领域母模型：", json.dumps(pilot["parent_audit"], indent=2, ensure_ascii=False))
        print("任务分区条数：", json.dumps(pilot["counts"], indent=2, ensure_ascii=False))
        print("当前 GPU：", torch.cuda.get_device_name(0))
        '''), code('''
        SFT_PROBE_SUCCEEDED = False
        SFT_TRAIN_SUCCEEDED = False
        SFT_BACKED_UP = False
        probe = run_sft_pilot(pilot, "probe", finllm)
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        SFT_PROBE_SUCCEEDED = True
        '''), code('''
        SFT_TRAIN_SUCCEEDED = False
        SFT_BACKED_UP = False
        if not globals().get("SFT_PROBE_SUCCEEDED", False):
            raise RuntimeError("请先完成 SFT 单步验证。")
        result = run_sft_pilot(pilot, "continue", finllm)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        SFT_TRAIN_SUCCEEDED = True
        '''), code('''
        SFT_EVAL_SUCCEEDED = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False):
            raise RuntimeError("请先完成SFT训练。")
        SFT_OUTPUT = Path(pilot["config"]["output_dir"])
        SFT_DEV = pilot["task_local"] / "development.jsonl"
        for name, checkpoint in (("domain_before", Path(pilot["config"]["init_checkpoint"])),
                                 ("sft_best", SFT_OUTPUT / "best.pt")):
            finllm("evaluate", "--checkpoint", checkpoint, "--tokenizer", pilot["config"]["tokenizer"],
                   "--input", SFT_DEV, "--output", SFT_OUTPUT / (name + "_development.json"),
                   "--target-mode", "literal_entity")
        SFT_EVAL_SUCCEEDED = True
        print("已保存同一开发集的领域模型与SFT最佳检查点评估。")
        '''), code('''
        SFT_BACKED_UP = False
        if not globals().get("SFT_TRAIN_SUCCEEDED", False) or not globals().get("SFT_EVAL_SUCCEEDED", False):
            raise RuntimeError("SFT训练或开发集比较尚未完成，保留现有检查点。")
        backup = backup_sft_pilot(pilot)
        print(json.dumps(backup, indent=2, ensure_ascii=False))
        SFT_BACKED_UP = True
        print("实体试验完成并备份：", pilot["stage_drive"] / "run")
        '''), markdown('''
        保留训练报告、development macro-F1 和 `verified: true` 备份输出；
        单次小样本数字可能波动。最终测试、资产标识映射、DPO偏好审核、校准和实际X场景评测尚待完成。
        ''')]
    write("04a_entity_sft_pilot.ipynb", cells)


if __name__ == "__main__":
    build()
