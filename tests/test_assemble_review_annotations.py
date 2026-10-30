import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("review_assembler", Path(__file__).resolve().parents[1] / "scripts/assemble_review_annotations.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_files(tmp_path):
    rows = [{"review_no": i, "id": f"row:{i}", "text": text, "source_entity": "ABC",
             "text_group_id": f"text:{i}", "source_symbol_candidates": ["ABC"],
             "mapping_status": "unique_historical_candidate", "human_reviewed": False,
             "human_stance": "", "asset_identifier": "", "original_200": True}
            for i, text in enumerate(("Buy ABC for growth", "ABC net profit increased"), 1)]
    source = tmp_path / "input.jsonl"
    source.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    annotation = tmp_path / "draft.tsv"
    annotation.write_text("1\tbullish\tBuy ABC\t明确建议买入。\tcompany\teasy\tmedium\n"
                          "2\tneutral\tnet profit increased\t仅报告利润增长。\tcompany\teasy\tlow\n", encoding="utf-8")
    return source, annotation, tmp_path / "output.jsonl"


def test_literal_proposals_remain_unapproved_and_rerun_is_readonly(tmp_path):
    source, annotation, output = fixture_files(tmp_path)
    rows, report = module.assemble(source, [annotation], output)
    before = output.stat().st_mtime_ns
    assert report["rows"] == 2 and report["human_reviewed_rows"] == 0
    assert all(not r["formal_training_ready"] and not r["human_asset_verified"] for r in rows)
    module.assemble(source, [annotation], output)
    assert output.stat().st_mtime_ns == before


@pytest.mark.parametrize("replacement", [
    "1\tbullish\tinvented quote\t说明。\tcompany\teasy\tlow\n",
    "1\tbullish\tBuy ABC\t说明。\tcompany\teasy\tlow\n" * 2,
    "9\tbullish\tBuy ABC\t说明。\tcompany\teasy\tlow\n",
])
def test_invented_evidence_duplicate_and_unknown_ids_fail_before_export(tmp_path, replacement):
    source, annotation, output = fixture_files(tmp_path)
    annotation.write_text(replacement, encoding="utf-8")
    with pytest.raises(ValueError):
        module.assemble(source, [annotation], output)
    assert not output.exists()


def test_missing_row_and_forged_human_review_rejected(tmp_path):
    source, annotation, output = fixture_files(tmp_path)
    annotation.write_text(annotation.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Every source"):
        module.assemble(source, [annotation], output)
    source.write_text(source.read_text(encoding="utf-8").replace('"human_reviewed": false', '"human_reviewed": true'), encoding="utf-8")
    with pytest.raises(ValueError, match="Preserve existing human"):
        module.assemble(source, [annotation], output)


def test_existing_edits_are_not_overwritten(tmp_path):
    source, annotation, output = fixture_files(tmp_path)
    output.write_text("human work preserved", encoding="utf-8")
    with pytest.raises(ValueError, match="preserved"):
        module.assemble(source, [annotation], output)
    assert output.read_text(encoding="utf-8") == "human work preserved"


def test_excel_csv_does_not_execute_formula_and_retains_raw_json(tmp_path):
    source, annotation, output = fixture_files(tmp_path)
    rows, _ = module.assemble(source, [annotation], output)
    rows[0]["text"] = '=HYPERLINK("example")'
    csv_path = tmp_path / "review.csv"
    module.export_csv(rows, csv_path)
    assert "'=HYPERLINK" in csv_path.read_text(encoding="utf-8-sig")
    assert json.loads(output.read_text().splitlines()[0])["text"] == "Buy ABC for growth"
