"""No network: check source allowlists, dates, byte caps, and rights-marker rejection."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("fetch_seed_corpora", Path(__file__).parents[1] / "scripts/fetch_seed_corpora.py")
seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(seed)


def test_calendar_uses_real_same_host_links_and_release_dates():
    pytest.importorskip("bs4", reason="HTML ingestion requires the optional data dependencies")
    html = b'''<div><a href="/monetarypolicy/fomcminutes20241218.htm">HTML</a>
    (Released January 08, 2025)</div>
    <a href="https://evil.test/monetarypolicy/fomcminutes20240101.htm">HTML</a>
    <a href="/monetarypolicy/fomcminutes20250129.htm">future set</a>'''
    found = seed.discover_minutes(html, {2024})
    assert found == [{"url": seed.FED_ORIGIN + "/monetarypolicy/fomcminutes20241218.htm",
                      "meeting_date": "2024-12-18", "published_at": "2025-01-08"}]
    assert not seed.allowed_fed_url("https://www.federalreserve.gov.evil.test/a")
    assert not seed.allowed_fed_url("http://www.federalreserve.gov/a")


def test_extract_uses_article_only_and_rejects_copyright():
    pytest.importorskip("bs4", reason="HTML ingestion requires the optional data dependencies")
    body = "Inflation was 2.5 percent and the outlook was uncertain. " * 30
    html = f'<nav>outside</nav><div id="article"><h3>FOMC minutes</h3><p>{body}<sup>1</sup></p><script>unwanted</script><a>Return to text</a></div>'
    title, text = seed.extract_minutes(html)
    assert title == "FOMC minutes"
    assert "2.5 percent" in text
    assert "outside" not in text and "unwanted" not in text and "Return to text" not in text
    with pytest.raises(ValueError, match="copyright marker"):
        seed.extract_minutes(html.replace(body, body + "Copyright third party"))


def test_general_output_is_byte_bounded_and_attributed(tmp_path, monkeypatch):
    rows = [{"id": str(i), "title": "Economics", "url": "https://en.wikipedia.org/wiki/Economics",
             "text": "An attributed article about economics. " * 20} for i in range(5)]
    fake = SimpleNamespace(load_dataset=lambda *a, **kw: iter(rows))
    monkeypatch.setitem(sys.modules, "datasets", fake)
    args = SimpleNamespace(retrieved_at="2026-10-09T00:00:00Z", general_max_bytes=3000,
                           max_general_documents=10)
    report = seed.fetch_general(args, tmp_path)
    content = (tmp_path / "general_candidate.jsonl").read_bytes()
    assert 0 < len(content) <= args.general_max_bytes
    assert report["sha256"] == seed.sha256(content)
    actual = [json.loads(line) for line in content.splitlines()]
    assert actual[0]["source_url"] == rows[0]["url"]
    assert actual[0]["source_revision"] == seed.WIKI_REVISION
    assert actual[0]["candidate_only"] is True
    with pytest.raises(FileExistsError):
        seed.fetch_general(args, tmp_path)


def test_existing_output_is_not_overwritten(tmp_path):
    (tmp_path / "precious.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        seed.main(["--output", str(tmp_path)])
    assert (tmp_path / "precious.txt").read_text() == "keep"


@pytest.fixture
def complete_download(tmp_path, monkeypatch):
    def general(args, output):
        payload = b'{"text":"General fixture"}\n'
        (output / "general_candidate.jsonl").write_bytes(payload)
        return {"file": "general_candidate.jsonl", "documents": 1, "bytes": len(payload),
                "sha256": seed.sha256(payload), "dataset": seed.WIKI_DATASET,
                "revision": seed.WIKI_REVISION, "config": seed.WIKI_CONFIG}

    def finance(args, output):
        payload = b'{"text":"Finance fixture"}\n'
        (output / "finance_candidate.jsonl").write_bytes(payload)
        raw = output / "source_snapshots"
        raw.mkdir()
        (raw / "fomc_calendar.html").write_bytes(b"calendar")
        (raw / "2024-01-01.html").write_bytes(b"article")
        seed.write_json(raw / "discovered_minutes.json", [{"meeting_date": "2024-01-01"}])
        return {"file": "finance_candidate.jsonl", "documents": 1, "bytes": len(payload),
                "sha256": seed.sha256(payload), "calendar_url": seed.FED_CALENDAR,
                "calendar_sha256": seed.sha256(b"calendar"), "years": args.finance_years,
                "discovered": 1, "admitted": [{"html_snapshot": "source_snapshots/2024-01-01.html",
                                               "html_sha256": seed.sha256(b"article")} ]}

    monkeypatch.setattr(seed, "fetch_general", general)
    monkeypatch.setattr(seed, "fetch_finance", finance)
    output = tmp_path / "corpora"
    assert seed.main(["--output", str(output)]) == 0

    def forbid(*args, **kwargs):
        pytest.fail("Existing output must be validated without refetching")

    monkeypatch.setattr(seed, "fetch_general", forbid)
    monkeypatch.setattr(seed, "fetch_finance", forbid)
    return output


def test_completed_download_rerun_is_read_only(complete_download, capsys):
    output = complete_download
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert seed.main(["--output", str(output)]) == 0
    assert '"reused": true' in capsys.readouterr().out
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


@pytest.mark.parametrize("change", ["incomplete", "missing_manifest", "candidate", "snapshot", "revision"])
def test_invalid_completed_download_requires_new_version(complete_download, capsys, change):
    output = complete_download
    manifest_path = output / "manifest.json"
    if change == "missing_manifest":
        manifest_path.unlink()
    elif change in {"incomplete", "revision"}:
        manifest = json.loads(manifest_path.read_text())
        if change == "incomplete":
            manifest["status"] = "incomplete"
        else:
            manifest["general"]["revision"] = "different"
        seed.write_json(manifest_path, manifest)
    else:
        changed = output / ("general_candidate.jsonl" if change == "candidate" else "source_snapshots/2024-01-01.html")
        changed.write_bytes(b"modified")
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    with pytest.raises(SystemExit):
        seed.main(["--output", str(output)])
    assert "use a new seed version" in capsys.readouterr().err
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize("options", [["--general-max-bytes", "30000000"], ["--finance-years", "2024"]])
def test_download_config_change_is_not_silently_reused(complete_download, options, capsys):
    with pytest.raises(SystemExit):
        seed.main(["--output", str(complete_download), *options])
    assert "use a new seed version" in capsys.readouterr().err
