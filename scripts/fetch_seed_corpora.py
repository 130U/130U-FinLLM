"""Fetch a bounded, attributed Wikipedia/FOMC seed corpus; no labels are inferred.

Run from the repo: python scripts/fetch_seed_corpora.py --output data/raw/seed_v1
Requires datasets, requests and beautifulsoup4. Source review is recorded in
docs/research/corpus_sources_20261009.md. Outputs remain unvalidated candidates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_REVISION = "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
WIKI_CONFIG = "20231101.en"
FED_ORIGIN = "https://www.federalreserve.gov"
FED_CALENDAR = FED_ORIGIN + "/monetarypolicy/fomccalendars.htm"
FED_TERMS = FED_ORIGIN + "/disclaimer.htm"
WIKI_TERMS = "https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use/en#7._Licensing_of_Content"
MAX_RESPONSE_BYTES = 2_000_000
USER_AGENT = "130U-FinLLM/0.1 (bounded research corpus; https://github.com/130U/130U-FinLLM)"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(content):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def json_line(row):
    return (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def allowed_fed_url(url):
    parsed = urlparse(url)
    return (parsed.scheme == "https" and parsed.netloc == "www.federalreserve.gov"
            and not parsed.query and not parsed.fragment)


class FedFetcher:
    """Serial, size-bounded requests; redirects cannot leave the approved host."""

    def __init__(self):
        import requests
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.last_request = 0.0

    def get(self, url):
        import requests
        if not allowed_fed_url(url):
            raise ValueError("Refusing a non-approved Federal Reserve URL")
        for attempt in range(2):
            try:
                current = url
                for redirect in range(4):
                    if not allowed_fed_url(current):
                        raise ValueError("Refusing cross-host or non-HTTPS redirect")
                    time.sleep(max(0.0, 1.0 - (time.monotonic() - self.last_request)))
                    self.last_request = time.monotonic()
                    with self.session.get(current, timeout=(10, 30), stream=True,
                                          allow_redirects=False) as response:
                        if response.status_code in (301, 302, 303, 307, 308):
                            current = urljoin(current, response.headers.get("Location", ""))
                            if redirect == 3:
                                raise ValueError("Too many redirects")
                            continue
                        response.raise_for_status()
                        if "text/html" not in response.headers.get("Content-Type", "").lower():
                            raise ValueError("Expected an HTML page")
                        if int(response.headers.get("Content-Length", "0")) > MAX_RESPONSE_BYTES:
                            raise ValueError("HTML response exceeds byte limit")
                        chunks, size = [], 0
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            if size > MAX_RESPONSE_BYTES:
                                raise ValueError("HTML response exceeds byte limit")
                            chunks.append(chunk)
                        return b"".join(chunks)
                raise ValueError("No final HTML response")
            except requests.RequestException:
                if attempt == 1:
                    raise
                time.sleep(1.0)
        raise RuntimeError("Unreachable request state")


def discover_minutes(calendar_html, years):
    """Read actual links and their published dates from FOMC calendar rows."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(calendar_html, "html.parser")
    found = {}
    for link in soup.find_all("a", href=True):
        url = urljoin(FED_CALENDAR, link["href"])
        parsed = urlparse(url)
        match = re.fullmatch(r"/monetarypolicy/fomcminutes(\d{8})\.htm", parsed.path)
        if not match or not allowed_fed_url(url) or int(match[1][:4]) not in years:
            continue
        published = None
        # Fed places the release date in the same minutes div or its parent.
        for parent in list(link.parents)[:4]:
            text = parent.get_text(" ", strip=True)
            release = re.search(r"Released\s+([A-Za-z]+\s+\d{1,2},\s+\d{4})", text)
            if release:
                published = datetime.strptime(release[1], "%B %d, %Y").date().isoformat()
                break
        date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
        found[url] = {"url": url, "meeting_date": date, "published_at": published}
    return sorted(found.values(), key=lambda item: item["meeting_date"])


def extract_minutes(html):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("#article")
    if article is None:
        raise ValueError("Expected #article was absent")
    title_element = article.find(["h1", "h2", "h3"])
    title = title_element.get_text(" ", strip=True) if title_element else "FOMC minutes"
    for element in article.select("script, style, nav, footer, button, sup"):
        element.decompose()
    for element in article.find_all("a"):
        if element.get_text(" ", strip=True).lower() in {"return to text", "back to top"}:
            element.decompose()
    text = "\n".join(line.strip() for line in article.get_text("\n").splitlines() if line.strip())
    if re.search(r"(?:copyright|all rights reserved|©)", text, flags=re.IGNORECASE):
        raise ValueError("Article contains an explicit copyright marker; separate review required")
    if len(text) < 1000:
        raise ValueError("Unexpectedly short FOMC article")
    return title, text


def fetch_general(args, output):
    from datasets import load_dataset
    source = load_dataset(WIKI_DATASET, name=WIKI_CONFIG, split="train", revision=WIKI_REVISION,
                          streaming=True, columns=["id", "url", "title", "text"])
    count, written, skipped = 0, 0, 0
    path = output / "general_candidate.jsonl"
    digest = hashlib.sha256()
    with path.open("xb") as stream:
        for article in source:
            text = article.get("text")
            if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 1_000_000:
                skipped += 1
                continue
            article_url = article.get("url")
            if not isinstance(article_url, str) or urlparse(article_url).netloc != "en.wikipedia.org":
                skipped += 1
                continue
            row = {"id": "wikipedia-en-" + str(article["id"]), "text": text,
                   "title": article["title"], "source_url": article_url, "split": "train",
                   "source": "https://huggingface.co/datasets/" + WIKI_DATASET,
                   "source_revision": WIKI_REVISION, "source_config": WIKI_CONFIG,
                   "license": ["CC-BY-SA-3.0", "GFDL"], "license_url": WIKI_TERMS,
                   "rights_note": "Wikipedia contributors; article URL provides attribution. Snapshot card states CC-BY-SA-3.0/GFDL; current Wikimedia terms state CC-BY-SA-4.0. Preserve applicable attribution/share-alike and extra source notices; see docs/research/corpus_sources_20261009.md.",
                   "training_allowed": True, "synthetic": False, "text_sha256": sha256(text),
                   "retrieved_at": args.retrieved_at, "candidate_only": True}
            encoded = json_line(row)
            if written + len(encoded) > args.general_max_bytes:
                break
            stream.write(encoded)
            digest.update(encoded)
            written += len(encoded)
            count += 1
            if count >= args.max_general_documents:
                break
    if count == 0:
        raise ValueError("No Wikipedia documents were admitted")
    return {"file": path.name, "documents": count, "bytes": written, "sha256": digest.hexdigest(),
            "skipped": skipped, "dataset": WIKI_DATASET, "revision": WIKI_REVISION,
            "config": WIKI_CONFIG, "selection": "bounded prefix; not a representative random sample"}


def fetch_finance(args, output, fetcher=None):
    fetcher = fetcher or FedFetcher()
    raw = output / "source_snapshots"
    raw.mkdir()
    calendar = fetcher.get(FED_CALENDAR)
    (raw / "fomc_calendar.html").write_bytes(calendar)
    minutes = discover_minutes(calendar, set(args.finance_years))
    if not minutes:
        raise ValueError("Calendar contained no minutes for the requested years")
    write_json(raw / "discovered_minutes.json", minutes)
    count, written, digest, skipped, admitted = 0, 0, hashlib.sha256(), [], []
    path = output / "finance_candidate.jsonl"
    with path.open("xb") as stream:
        for item in minutes:
            if count >= args.max_finance_documents:
                break
            if not item["published_at"]:
                skipped.append({**item, "reason": "public release date not found in calendar"})
                continue
            html = fetcher.get(item["url"])
            try:
                title, text = extract_minutes(html)
            except ValueError as error:
                skipped.append({**item, "reason": str(error), "html_sha256": sha256(html)})
                continue
            row = {"id": "fomc-" + item["meeting_date"], "text": text, "title": title,
                   "source": item["url"], "source_url": item["url"], "split": "train",
                   "meeting_date": item["meeting_date"], "published_at": item["published_at"],
                   "retrieved_at": args.retrieved_at, "license": "public-domain-US-government",
                   "license_url": FED_TERMS, "training_allowed": True, "synthetic": False,
                   "rights_note": "Federal Reserve Board original FOMC minutes. Board disclaimer states public domain unless otherwise indicated; Board attribution retained. Explicit article copyright markers excluded; third-party images and external linked content not ingested.",
                   "html_sha256": sha256(html), "text_sha256": sha256(text),
                   "extraction": "#article; navigation/scripts/superscripts/return links removed; whitespace normalized",
                   "candidate_only": True}
            encoded = json_line(row)
            if written + len(encoded) > args.finance_max_bytes:
                break
            stream.write(encoded)
            digest.update(encoded)
            written += len(encoded)
            count += 1
            snapshot = item["meeting_date"] + ".html"
            (raw / snapshot).write_bytes(html)
            admitted.append({**item, "html_snapshot": "source_snapshots/" + snapshot,
                             "html_sha256": sha256(html), "text_sha256": sha256(text)})
    if count == 0:
        raise ValueError("No FOMC documents were admitted")
    return {"file": path.name, "documents": count, "bytes": written, "sha256": digest.hexdigest(),
            "calendar_url": FED_CALENDAR, "calendar_sha256": sha256(calendar), "years": args.finance_years,
            "discovered": len(minutes), "admitted": admitted, "skipped": skipped,
            "note": "Small macroeconomic corpus; contains no asset-stance labels or measured token count"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--general-max-bytes", type=int, default=20_000_000)
    parser.add_argument("--max-general-documents", type=int, default=10_000)
    parser.add_argument("--finance-years", default="2021,2022,2023,2024")
    parser.add_argument("--max-finance-documents", type=int, default=32)
    parser.add_argument("--finance-max-bytes", type=int, default=5_000_000)
    args = parser.parse_args(argv)
    for name in ("general_max_bytes", "max_general_documents", "max_finance_documents", "finance_max_bytes"):
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    try:
        args.finance_years = sorted(set(int(year) for year in args.finance_years.split(",")))
    except ValueError:
        parser.error("finance-years must contain comma-separated years")
    if not args.finance_years or any(year < 1900 or year > datetime.now().year for year in args.finance_years):
        parser.error("finance-years contains an invalid or future year")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be a new or empty directory")
    args.output.mkdir(parents=True, exist_ok=True)
    args.retrieved_at = utc_now()
    manifest = {"schema_version": 1, "status": "in_progress", "retrieved_at": args.retrieved_at,
                "extractor_sha256": sha256(Path(__file__).read_bytes()),
                "rights_review": "docs/research/corpus_sources_20261009.md",
                "limits": {name: getattr(args, name) for name in
                           ("general_max_bytes", "max_general_documents", "max_finance_documents", "finance_max_bytes")},
                "candidates_only": True, "note": "Not yet deduplicated, split, tokenized or checked against frozen evaluation texts"}
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    try:
        manifest["general"] = fetch_general(args, args.output)
        write_json(manifest_path, manifest)
        print(json.dumps({"general_documents": manifest["general"]["documents"],
                          "general_bytes": manifest["general"]["bytes"]}), flush=True)
        manifest["finance"] = fetch_finance(args, args.output)
        manifest["status"] = "complete"
        manifest["completed_at"] = utc_now()
    except Exception as error:
        manifest["status"] = "incomplete"
        manifest["error"] = type(error).__name__ + ": " + str(error)
        write_json(manifest_path, manifest)
        raise
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "complete", "manifest": str(manifest_path),
                      "general_documents": manifest["general"]["documents"],
                      "finance_documents": manifest["finance"]["documents"],
                      "total_candidate_bytes": manifest["general"]["bytes"] + manifest["finance"]["bytes"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
