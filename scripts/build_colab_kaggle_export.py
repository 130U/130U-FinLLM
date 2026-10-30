"""Generate a CPU-only, user-controlled Colab-to-Kaggle transfer notebook."""
from build_notebooks import BOOTSTRAP, code, markdown, write
from build_expansion_notebooks import DRIVE


def build():
    cells = [markdown('''
        # 04b｜用 Colab CPU 准备 Kaggle 私有转运包

        本页读取你的 Drive 快照，核对领域检查点、8K词表及实际预训练排除清单，
        将需要的原始已验证快照流式打成 ZIP。无需 GPU，不训练也不修改 Drive 中的模型。

        在 Colab 选择「运行时 → 更改运行时类型 → 无」，从第一格顺序运行并挂载自己的 Drive。
        任务包、用户标注和模型均为私人材料，后续仅上传为 Kaggle **Private** 数据集。
        转运包保存在临时的 `/content`，断线前请下载到电脑。实际大小以输出为准，可能超过 1 GB；
        完成后记下 ZIP SHA256 和清单 manifest_sha256。不要同时运行另一份训练或修改原快照。
        '''), code(BOOTSTRAP), code(DRIVE), code('''
        from finllm.kaggle_export import export_kaggle_bundle
        TRANSFER_ZIP = Path("/content/130u_kaggle_transfer.zip")
        export_report = export_kaggle_bundle(DRIVE_ROOT, TRANSFER_ZIP)
        print("Kaggle 私有转运包已校验：", json.dumps(export_report, indent=2, ensure_ascii=False))
        print("请记录 ZIP SHA256：", export_report["sha256"])
        print("请记录清单 manifest_sha256（Kaggle解压后使用）：", export_report["manifest_sha256"])
        '''), code('''
        from google.colab import files
        if not TRANSFER_ZIP.is_file():
            raise FileNotFoundError("请先完成上一格的转运包生成与校验。")
        files.download(str(TRANSFER_ZIP))
        print("浏览器下载已发起。请等待下载完成，再上传到私有 Kaggle 数据集。")
        '''), markdown('''
        本页不启动 SFT，不会保证 Kaggle 已接收此包。若数据校验失败或空间不足，请保留完整报错。
        如果中断后存在同名临时文件，选择新的 ZIP 文件名；不要覆盖或删除 Drive 原件。
        ''')]
    write("04b_kaggle_export.ipynb", cells)


if __name__ == "__main__":
    build()
