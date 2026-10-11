"""Offline regression coverage for bounded expansion and interruption recovery."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("fetch_expansion_corpora", Path(__file__).parents[1] / "scripts/fetch_expansion_corpora.py")
expansion = importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)


def args_for(tmp_path, **changes):
    values = dict(output=tmp_path, retrieved_at="2026-10-11T00:00:00Z", general_max_bytes=100_000,
                  max_general_documents=5, finance_years=[2011, 2025],
                  max_finance_documents=2, finance_max_bytes=50_000)
    values.update(changes)
    (tmp_path / ".work").mkdir(exist_ok=True)
    return SimpleNamespace(**values)


def articles():
    return [{"id": str(i), "title": "Article " + str(i),
             "url": "https://en.wikipedia.org/wiki/Article_" + str(i),
             "text": ("Scientific and financial information in a complete attributed article. " + str(i)) * 30}
            for i in range(8)]


def install_dataset(monkeypatch, source):
    def load(*a, **kw):
        assert kw["revision"] == expansion.seed.WIKI_REVISION
        assert kw["streaming"] is True
        return source()
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load))


def test_general_reuses_complete_section_without_network(tmp_path, monkeypatch):
    install_dataset(monkeypatch, lambda: iter(articles()))
    args = args_for(tmp_path)
    report = expansion.fetch_general(args, tmp_path)
    assert report["documents"] == 5
    assert report["bytes"] <= args.general_max_bytes
    rows = [json.loads(line) for line in (tmp_path / report["file"]).read_text().splitlines()]
    assert rows[0]["source_url"].startswith("https://en.wikipedia.org/")
    assert rows[0]["candidate_only"] is True
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob("*") if p.is_file()}
    install_dataset(monkeypatch, lambda: pytest.fail("Completed section must not refetch"))
    assert expansion.fetch_general(args, tmp_path) == report
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_general_recovers_committed_rows_and_discards_only_uncommitted_tail(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion, "PROGRESS_EVERY", 2)
    def interrupted():
        yield from articles()[:3]
        raise ConnectionError("interrupted stream")
    install_dataset(monkeypatch, interrupted)
    args = args_for(tmp_path)
    with pytest.raises(ConnectionError):
        expansion.fetch_general(args, tmp_path)
    progress = json.loads((tmp_path / ".work/general_progress.json").read_text())
    assert progress["source_rows"] == 2
    committed = (tmp_path / ".work/general_candidate.jsonl.part").read_bytes()[:progress["bytes"]]
    install_dataset(monkeypatch, lambda: iter(articles()))
    report = expansion.fetch_general(args, tmp_path)
    actual = (tmp_path / report["file"]).read_bytes()
    assert actual.startswith(committed)
    assert [json.loads(line)["id"] for line in actual.splitlines()] == ["wikipedia-en-" + str(i) for i in range(5)]


def test_general_rejects_changed_committed_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(expansion, "PROGRESS_EVERY", 2)
    def interrupted():
        yield from articles()[:2]
        raise ConnectionError("interrupted")
    install_dataset(monkeypatch, interrupted)
    args = args_for(tmp_path)
    with pytest.raises(ConnectionError):
        expansion.fetch_general(args, tmp_path)
    partial = tmp_path / ".work/general_candidate.jsonl.part"
    payload = partial.read_bytes()
    partial.write_bytes(b"!" + payload[1:])
    with pytest.raises(ValueError, match="Committed Wikipedia progress has changed"):
        expansion.fetch_general(args, tmp_path)
    assert partial.read_bytes() == b"!" + payload[1:]


def test_section_recovers_marker_to_final_rename_interruption(tmp_path, monkeypatch):
    install_dataset(monkeypatch, lambda: iter(articles()))
    args = args_for(tmp_path)
    replace = expansion.os.replace
    def interrupted_replace(source, destination):
        if str(destination).endswith("general_candidate.jsonl"):
            raise OSError("interrupted rename")
        return replace(source, destination)
    monkeypatch.setattr(expansion.os, "replace", interrupted_replace)
    with pytest.raises(OSError):
        expansion.fetch_general(args, tmp_path)
    monkeypatch.setattr(expansion.os, "replace", replace)
    assert expansion.fetch_general(args, tmp_path)["documents"] == 5


def test_historical_discovery_follows_only_actual_same_host_links():
    html = b'''<a href="/monetarypolicy/fomchistorical2011.htm">2011</a>
        <a href="https://evil.test/monetarypolicy/fomchistorical2012.htm">2012</a>
        <a href="/monetarypolicy/fomchistorical2013.htm?bad=1">2013</a>'''
    assert expansion.discover_historical_years(html, {2011, 2012, 2013}) == {
        2011: expansion.seed.FED_ORIGIN + "/monetarypolicy/fomchistorical2011.htm"}


BODY = "Inflation and employment remained uncertain. Monetary policy and financial conditions were discussed. " * 30


def legacy_article(extra=""):
    return (f'<nav>outside</nav><div id="printThis" class="generalContentText">'
            f'<h1>Minutes of the Federal Open Market Committee</h1><div id="leftText"><p>{BODY}{extra}</p></div>'
            '<div id="rightText">external linked sidebar</div><div id="printIcon">Print</div></div>').encode()


def test_legacy_extraction_is_narrow_and_rejects_copyright():
    title, text, selector = expansion.extract_minutes(legacy_article())
    assert "Open Market Committee" in title and "#printThis" in selector
    assert "outside" not in text and "sidebar" not in text and "Print" not in text
    with pytest.raises(ValueError, match="copyright marker"):
        expansion.extract_minutes(legacy_article("Copyright third party"))
    with pytest.raises(ValueError, match="reviewed legacy"):
        expansion.extract_minutes(b"<div id='content'>" + BODY.encode() + b"</div>")


class FixtureFetcher:
    def __init__(self, interrupt=False):
        self.calls = []
        self.interrupt = interrupt

    def get(self, url):
        self.calls.append(url)
        if url == expansion.seed.FED_CALENDAR:
            return b'<div><a href="/monetarypolicy/fomcminutes20251210.htm">HTML</a>(Released December 30, 2025)</div>'
        if url == expansion.FED_HISTORY:
            return b'<a href="/monetarypolicy/fomchistorical2011.htm">2011</a>'
        if url.endswith("fomchistorical2011.htm"):
            return b'<p>Minutes (Released February 16, 2011):<a href="/monetarypolicy/fomcminutes20110126.htm">HTML</a></p>'
        if url.endswith("fomcminutes20110126.htm"):
            return legacy_article()
        if self.interrupt:
            raise ConnectionError("interrupt second article")
        return ('<div id="article"><h3>FOMC minutes</h3><p>' + BODY + '</p></div>').encode()


def test_finance_recovers_cached_pages_after_interruption(tmp_path):
    args = args_for(tmp_path)
    first = FixtureFetcher(interrupt=True)
    with pytest.raises(ConnectionError):
        expansion.fetch_finance(args, tmp_path, first)
    second = FixtureFetcher()
    report = expansion.fetch_finance(args, tmp_path, second)
    assert second.calls == [expansion.seed.FED_ORIGIN + "/monetarypolicy/fomcminutes20251210.htm"]
    assert report["documents"] == 2
    rows = [json.loads(line) for line in (tmp_path / report["file"]).read_text().splitlines()]
    assert rows[0]["published_at"] == "2011-02-16"
    assert rows[0]["source_url"].endswith("fomcminutes20110126.htm")
    expansion.validate_finance_snapshots(tmp_path, report)


def test_finance_rejects_changed_cache_before_refetch(tmp_path):
    args = args_for(tmp_path)
    with pytest.raises(ConnectionError):
        expansion.fetch_finance(args, tmp_path, FixtureFetcher(interrupt=True))
    cache = next((tmp_path / ".work/fed_cache").glob("*.json"))
    item = json.loads(cache.read_text())
    item["sha256"] = "changed"
    expansion.atomic_json(cache, item)
    # Discovery is already complete, so specifically exercise that cached page.
    with pytest.raises(ValueError, match="Cached Federal Reserve page has changed"):
        expansion.cached_html(tmp_path, FixtureFetcher(), item["url"], "a.html")


def test_main_complete_rerun_is_readonly_and_limits_changes_are_rejected(tmp_path, monkeypatch):
    install_dataset(monkeypatch, lambda: iter(articles()))
    monkeypatch.setattr(expansion.seed, "FedFetcher", FixtureFetcher)
    output = tmp_path / "corpora"
    options = ["--output", str(output), "--general-max-bytes", "100000", "--max-general-documents", "5",
               "--finance-years", "2011,2025", "--max-finance-documents", "2", "--finance-max-bytes", "50000"]
    assert expansion.main(options) == 0
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.rglob("*") if p.is_file()}
    assert expansion.main(options) == 0
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}
    with pytest.raises(SystemExit):
        expansion.main(options + ["--general-max-bytes", "200000"])
    assert before == {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before}


def test_completed_manifest_detects_article_snapshot_changes(tmp_path, monkeypatch):
    install_dataset(monkeypatch, lambda: iter(articles()))
    monkeypatch.setattr(expansion.seed, "FedFetcher", FixtureFetcher)
    options = ["--output", str(tmp_path / "corpora"), "--finance-years", "2011,2025"]
    expansion.main(options)
    snapshot = tmp_path / "corpora/source_snapshots/2011-01-26.html"
    snapshot.write_bytes(b"changed")
    with pytest.raises(ValueError, match="snapshot has changed"):
        expansion.main(options)


def test_unrecognized_existing_output_is_not_overwritten(tmp_path):
    precious = tmp_path / "precious.txt"
    precious.write_text("keep")
    with pytest.raises(SystemExit):
        expansion.main(["--output", str(tmp_path)])
    assert precious.read_text() == "keep"
