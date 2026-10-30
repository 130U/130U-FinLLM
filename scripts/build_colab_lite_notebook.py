"""Generate a small, CPU-only transfer of the user's already trained weights."""
from build_notebooks import BOOTSTRAP, markdown, code, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04d｜用 CPU 导出 Kaggle 轻量模型包

        从 Drive 已验证的 `domain_v1` 74步检查点提取模型权重，另带原8K词表与已审核任务数据。
        不需要旧的1.54 GiB转运包；原Drive文件不修改。本页不训练，选择 CPU 后从第一格运行。
        先核验领域检查点SHA、词表和任务清单，再下载约100 MB轻量包；实际大小以输出为准。

        之前04b的云端结果已经报告四份排除池覆盖全部1008条标题；本轻量包保留这一报告证据，
        不携带所有历史语料，因此Kaggle不能重新独立核验这些历史排除池。
        包含私人模型与新闻标题，只能放进你自己的 Kaggle Private 数据集。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.kaggle_lite import export_lite_from_drive
        LITE_ZIP = Path("/content/130u_kaggle_lite.zip")
        lite_report = export_lite_from_drive(DRIVE_ROOT, LITE_ZIP)
        print("轻量模型包已生成：", json.dumps(lite_report, indent=2, ensure_ascii=False))
        print("请记录 ZIP SHA256：", lite_report["sha256"])
        print("请记录 manifest_sha256：", lite_report["manifest_sha256"])
        '''), code('''
        from google.colab import files
        if not LITE_ZIP.is_file():
            raise FileNotFoundError("请先完成轻量模型包生成。")
        files.download(str(LITE_ZIP))
        print("浏览器下载已发起；请等待下载完成后上传到 Kaggle Private 数据集。")
        '''), markdown('''
        迁移只保留模型权重，不迁移旧优化器状态。Kaggle 将用新优化器开始 SFT。
        如需再次执行，请先确认已下载的文件安全，使用新文件名；原Drive成果保持原样。
        ''')]
    write("04d_colab_lite_export.ipynb", cells)


if __name__ == "__main__":
    build()
