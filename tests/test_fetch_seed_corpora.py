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
