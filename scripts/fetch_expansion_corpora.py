"""Fetch bounded Wikipedia/FOMC expansion candidates, preserving resumable progress.

Completed outputs are immutable. Partial Wikipedia progress and verified Federal
Reserve page caches resume after interruption; neither source infers stance labels.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

# This also works when imported directly by a notebook or an offline test.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_seed_corpora as seed

FED_HISTORY = seed.FED_ORIGIN + "/monetarypolicy/fomc_historical_year.htm"
SOURCE_REVIEW = "docs/research/expanded_corpus_sources_20261011.md"
PROGRESS_EVERY = 100
LIMIT_FIELDS = seed.LIMIT_FIELDS
CONTRACT_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contract(args):
    return {"contract_version": CONTRACT_VERSION,
            "limits": {name: getattr(args, name) for name in LIMIT_FIELDS},
            "finance_years": args.finance_years, "wiki_dataset": seed.WIKI_DATASET,
            "wiki_revision": seed.WIKI_REVISION, "wiki_config": seed.WIKI_CONFIG,
            "fed_calendar": seed.FED_CALENDAR, "fed_history": FED_HISTORY}


def verify_file(path, metadata):
    if Path(path).stat().st_size != metadata["bytes"] or seed.file_sha256(path) != metadata["sha256"]:
        raise ValueError(f"File integrity changed: {path}")


def completed_section(output, name):
    marker = output / ".work" / (name + "_complete.json")
    path = output / (name + "_candidate.jsonl")
    part = output / ".work" / (name + "_candidate.jsonl.part")
    if not marker.exists():
        if path.exists():
            raise ValueError("Candidate file exists without a verified section marker")
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if metadata.get("file") != path.name or metadata.get("documents", 0) < 1:
        raise ValueError("Invalid section completion marker")
    # A crash between writing the marker and final atomic rename is recoverable.
    verify_file(path if path.exists() else part, metadata)
    if not path.exists():
        os.replace(part, path)
    return metadata


def finish_section(output, name, part, metadata):
    atomic_json(output / ".work" / (name + "_complete.json"), metadata)
    os.replace(part, output / metadata["file"])
    return metadata


def wikipedia_row(article, retrieved_at):
    text = article.get("text")
    url = article.get("url")
    if (not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 1_000_000
            or not isinstance(url, str) or urlparse(url).netloc != "en.wikipedia.org"):
        return None
    return {"id": "wikipedia-en-" + str(article["id"]), "text": text,
            "title": article["title"], "source_url": url, "split": "train",
            "source": "https://huggingface.co/datasets/" + seed.WIKI_DATASET,
            "source_revision": seed.WIKI_REVISION, "source_config": seed.WIKI_CONFIG,
            "license": ["CC-BY-SA-3.0", "GFDL"], "license_url": seed.WIKI_TERMS,
            "rights_note": "Wikipedia contributors; article URL provides attribution. Snapshot card states CC-BY-SA-3.0/GFDL; current Wikimedia terms state CC-BY-SA-4.0. Preserve applicable attribution/share-alike and extra source notices; see " + SOURCE_REVIEW,
            "training_allowed": True, "synthetic": False, "text_sha256": seed.sha256(text),
            "retrieved_at": retrieved_at, "candidate_only": True}


def fetch_general(args, output):
    cached = completed_section(output, "general")
    if cached:
        return cached
    from datasets import load_dataset
    work = output / ".work"
    part = work / "general_candidate.jsonl.part"
    journal = work / "general_progress.json"
    state = {"source_rows": 0, "documents": 0, "bytes": 0, "skipped": 0,
             "sha256": hashlib.sha256().hexdigest()}
    if journal.exists():
        state = json.loads(journal.read_text(encoding="utf-8"))
        digest = hashlib.sha256()
        with part.open("rb") as stream:
            remaining = state["bytes"]
            while remaining:
                chunk = stream.read(min(1_048_576, remaining))
                if not chunk:
                    raise ValueError("Wikipedia progress file is shorter than its journal")
                digest.update(chunk)
                remaining -= len(chunk)
        if digest.hexdigest() != state["sha256"]:
            raise ValueError("Committed Wikipedia progress has changed; preserve this version")
        # Only uncommitted bytes after the checked journal boundary are discarded.
        with part.open("r+b") as stream:
            stream.truncate(state["bytes"])
    else:
        if part.exists():
            raise ValueError("Wikipedia partial file has no progress journal")
        digest = hashlib.sha256()
        part.touch(exist_ok=False)
        atomic_json(journal, state)
    source = load_dataset(seed.WIKI_DATASET, name=seed.WIKI_CONFIG, split="train",
                          revision=seed.WIKI_REVISION, streaming=True,
                          columns=["id", "url", "title", "text"])
    resumed_rows = state["source_rows"]
    with part.open("ab") as stream:
        def checkpoint():
            stream.flush()
            os.fsync(stream.fileno())
            state["sha256"] = digest.hexdigest()
            atomic_json(journal, state)

        for source_index, article in enumerate(source):
            if source_index < resumed_rows:
                continue
            if state["documents"] >= args.max_general_documents:
                break
            row = wikipedia_row(article, args.retrieved_at)
            encoded = seed.json_line(row) if row is not None else None
            if encoded is not None and state["bytes"] + len(encoded) > args.general_max_bytes:
                break
            state["source_rows"] = source_index + 1
            if encoded is None:
                state["skipped"] += 1
            else:
                stream.write(encoded)
                digest.update(encoded)
                state["bytes"] += len(encoded)
                state["documents"] += 1
            if state["source_rows"] % PROGRESS_EVERY == 0:
                checkpoint()
        checkpoint()
    if state["documents"] < 1:
        raise ValueError("No Wikipedia documents fit the configured limits")
    metadata = {**state, "file": "general_candidate.jsonl", "dataset": seed.WIKI_DATASET,
                "revision": seed.WIKI_REVISION, "config": seed.WIKI_CONFIG,
                "selection": "bounded source-order prefix; seed overlap is excluded during preparation; not a representative random sample"}
    return finish_section(output, "general", part, metadata)


def cached_html(output, fetcher, url, snapshot):
    """Each downloaded page has one atomically committed, self-checking cache."""
    cache = output / ".work" / "fed_cache" / (seed.sha256(url) + ".json")
    if cache.exists():
        item = json.loads(cache.read_text(encoding="utf-8"))
        html = base64.b64decode(item["html_base64"], validate=True)
        if item["url"] != url or len(html) != item["bytes"] or seed.sha256(html) != item["sha256"]:
            raise ValueError("Cached Federal Reserve page has changed")
    else:
        html = fetcher.get(url)
        atomic_json(cache, {"url": url, "bytes": len(html), "sha256": seed.sha256(html),
                            "html_base64": base64.b64encode(html).decode("ascii")})
    path = output / "source_snapshots" / snapshot
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != html:
            raise ValueError("Federal Reserve source snapshot has changed")
    else:
        temporary = path.with_name(path.name + ".writing")
        temporary.write_bytes(html)
        os.replace(temporary, path)
    return html, {"url": url, "file": path.relative_to(output).as_posix(),
                  "bytes": len(html), "sha256": seed.sha256(html)}


def discover_historical_years(index_html, years):
    from bs4 import BeautifulSoup
    found = {}
    for link in BeautifulSoup(index_html, "html.parser").find_all("a", href=True):
        url = urljoin(FED_HISTORY, link["href"])
        match = re.fullmatch(r"/monetarypolicy/fomchistorical(\d{4})\.htm", urlparse(url).path)
        if match and seed.allowed_fed_url(url) and int(match[1]) in years:
            found[int(match[1])] = url
    return found


def discover_finance(args, output, fetcher):
    marker = output / ".work" / "finance_discovery.json"
    if marker.exists():
        state = json.loads(marker.read_text(encoding="utf-8"))
        for page in state["snapshots"]:
            verify_file(output / page["file"], page)
        return state
    calendar, calendar_meta = cached_html(output, fetcher, seed.FED_CALENDAR, "fomc_calendar.html")
    minutes = {item["url"]: item for item in seed.discover_minutes(calendar, set(args.finance_years))}
    snapshots = [calendar_meta]
    present = {int(item["meeting_date"][:4]) for item in minutes.values()}
    missing = set(args.finance_years) - present
    if missing:
        index, index_meta = cached_html(output, fetcher, FED_HISTORY, "fomc_historical_year.html")
        snapshots.append(index_meta)
        years = discover_historical_years(index, missing)
        if missing - set(years):
            raise ValueError("Requested FOMC years absent from actual calendar/history links: " + str(sorted(missing - set(years))))
        for year, url in sorted(years.items()):
            page, metadata = cached_html(output, fetcher, url, f"fomc_history_{year}.html")
            snapshots.append(metadata)
            discovered = seed.discover_minutes(page, {year})
            if not discovered:
                raise ValueError(f"Historical calendar had no actual HTML minutes links for {year}")
            minutes.update({item["url"]: item for item in discovered})
    state = {"minutes": sorted(minutes.values(), key=lambda item: item["meeting_date"]),
             "snapshots": snapshots}
    if not state["minutes"]:
        raise ValueError("No actual FOMC HTML links were discovered")
    atomic_json(marker, state)
    atomic_json(output / "source_snapshots" / "discovered_minutes.json", state["minutes"])
    return state


def extract_minutes(html):
    """Admit the verified legacy print body as well as modern #article pages."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one("#article") is not None:
        title, text = seed.extract_minutes(html)
        return title, text, "#article"
    article = soup.select_one("#printThis.generalContentText")
    if article is None or article.select_one("#leftText") is None:
        raise ValueError("Expected modern or reviewed legacy FOMC article body was absent")
    heading = article.find("h1")
    if heading is None or "Minutes of the Federal Open Market Committee" not in heading.get_text(" ", strip=True):
        raise ValueError("Legacy print body was not a FOMC minutes article")
    title = heading.get_text(" ", strip=True)
    for element in article.select("script, style, nav, footer, button, sup, #printIcon, .contentNav, #rightText, img"):
        element.decompose()
    for element in article.find_all("a"):
        if element.get_text(" ", strip=True).lower() in {"return to text", "back to top"}:
            element.decompose()
    text = "\n".join(line.strip() for line in article.get_text("\n").splitlines() if line.strip())
    if re.search(r"(?:copyright|all rights reserved|©)", text, flags=re.IGNORECASE):
        raise ValueError("Article contains an explicit copyright marker; separate review required")
    if len(text) < 1000:
        raise ValueError("Unexpectedly short FOMC article")
    return title, text, "#printThis.generalContentText; verified FOMC h1 and #leftText"


def validate_finance_snapshots(output, metadata):
    for item in metadata["discovery_snapshots"]:
        verify_file(output / item["file"], item)
    for item in metadata["admitted"]:
        path = output / item["html_snapshot"]
        if not path.resolve().is_relative_to((output / "source_snapshots").resolve()):
            raise ValueError("Article snapshot escaped the source directory")
        if seed.file_sha256(path) != item["html_sha256"]:
            raise ValueError("Admitted Federal Reserve article snapshot has changed")


def fetch_finance(args, output, fetcher=None):
    cached = completed_section(output, "finance")
    if cached:
        validate_finance_snapshots(output, cached)
        return cached
    fetcher = fetcher or seed.FedFetcher()
    discovery = discover_finance(args, output, fetcher)
    part = output / ".work" / "finance_candidate.jsonl.part"
    digest, admitted, skipped, written = hashlib.sha256(), [], [], 0
    # Only an incomplete, unpublished section is rebuilt from immutable page caches.
    with part.open("wb") as stream:
        for item in discovery["minutes"]:
            if len(admitted) >= args.max_finance_documents:
                break
            if not item["published_at"]:
                skipped.append({**item, "reason": "public release date absent from source calendar"})
                continue
            snapshot = item["meeting_date"] + ".html"
            html, _ = cached_html(output, fetcher, item["url"], snapshot)
            try:
                title, text, selector = extract_minutes(html)
            except ValueError as error:
                skipped.append({**item, "reason": str(error), "html_sha256": seed.sha256(html)})
                continue
            row = {"id": "fomc-" + item["meeting_date"], "text": text, "title": title,
                   "source": item["url"], "source_url": item["url"], "split": "train",
                   "meeting_date": item["meeting_date"], "published_at": item["published_at"],
                   "retrieved_at": args.retrieved_at, "license": "public-domain-US-government",
                   "license_url": seed.FED_TERMS, "training_allowed": True, "synthetic": False,
                   "rights_note": "Federal Reserve Board original FOMC minutes; Board attribution retained. Board disclaimer states public domain unless otherwise indicated; explicit article copyright markers excluded. No third-party images or external linked content ingested.",
                   "html_sha256": seed.sha256(html), "text_sha256": seed.sha256(text),
                   "extraction": selector + "; navigation/scripts/superscripts/return links removed; whitespace normalized",
                   "candidate_only": True}
            encoded = seed.json_line(row)
            if written + len(encoded) > args.finance_max_bytes:
                break
            stream.write(encoded)
            digest.update(encoded)
            written += len(encoded)
            admitted.append({**item, "html_snapshot": "source_snapshots/" + snapshot,
                             "html_sha256": seed.sha256(html), "text_sha256": seed.sha256(text)})
        stream.flush()
        os.fsync(stream.fileno())
    if not admitted:
        raise ValueError("No FOMC documents fit the configured limits")
    metadata = {"file": "finance_candidate.jsonl", "documents": len(admitted),
                "bytes": written, "sha256": digest.hexdigest(), "calendar_url": seed.FED_CALENDAR,
                "history_url": FED_HISTORY, "years": args.finance_years,
                "discovered": len(discovery["minutes"]), "discovery_snapshots": discovery["snapshots"],
                "admitted": admitted, "skipped": skipped,
                "note": "Macro-finance language corpus; no stance labels; unique tokens must be measured after seed/evaluation exclusion"}
    return finish_section(output, "finance", part, metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--general-max-bytes", type=int, default=200_000_000)
    parser.add_argument("--max-general-documents", type=int, default=50_000)
    parser.add_argument("--finance-years", default=",".join(str(year) for year in range(2011, 2026)))
    parser.add_argument("--max-finance-documents", type=int, default=128)
    parser.add_argument("--finance-max-bytes", type=int, default=12_000_000)
    args = parser.parse_args(argv)
    for name in LIMIT_FIELDS:
        if getattr(args, name) < 1:
            parser.error(name + " must be positive")
    try:
        args.finance_years = sorted(set(int(year) for year in args.finance_years.split(",")))
    except ValueError:
        parser.error("finance-years must contain comma-separated integer years")
    if not args.finance_years or any(year < 2011 or year > min(2025, datetime.now().year) for year in args.finance_years):
        parser.error("finance-years must be within the reviewed completed meeting years 2011–2025")
    output, expected = args.output, contract(args)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != expected:
            parser.error("Existing expansion source or limits differ; preserve it and choose a new version")
        args.retrieved_at = manifest["retrieved_at"]
        if manifest.get("status") == "complete":
            for name in ("general", "finance"):
                section = manifest[name]
                verify_file(output / section["file"], section)
            validate_finance_snapshots(output, manifest["finance"])
            print(json.dumps({"status": "complete", "reused": True, "manifest": str(manifest_path),
                              "general_documents": manifest["general"]["documents"],
                              "finance_documents": manifest["finance"]["documents"]}, indent=2))
            return 0
    else:
        if output.exists() and any(output.iterdir()):
            parser.error("Nonempty output has no expansion manifest; preserve it and choose a new version")
        args.retrieved_at = seed.utc_now()
        output.mkdir(parents=True, exist_ok=True)
        manifest = {"schema_version": 1, "status": "in_progress", "retrieved_at": args.retrieved_at,
                    "contract": expected, "limits": expected["limits"], "rights_review": SOURCE_REVIEW,
                    "extractor_sha256": seed.sha256(Path(__file__).read_bytes()), "candidates_only": True,
                    "note": "No labels inferred. Preparation must exclude original seed/evaluation groups before tokenization."}
        atomic_json(manifest_path, manifest)
    (output / ".work").mkdir(exist_ok=True)
    try:
        for name, fetch in (("general", fetch_general), ("finance", fetch_finance)):
            manifest[name] = fetch(args, output)
            manifest["status"] = "in_progress"
            manifest.pop("error", None)
            atomic_json(manifest_path, manifest)
            print(json.dumps({"source": name, "documents": manifest[name]["documents"],
                              "bytes": manifest[name]["bytes"]}), flush=True)
        manifest["status"] = "complete"
        manifest["completed_at"] = seed.utc_now()
        atomic_json(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "incomplete"
        manifest["error"] = type(error).__name__ + ": " + str(error)
        atomic_json(manifest_path, manifest)
        raise
    print(json.dumps({"status": "complete", "manifest": str(manifest_path),
                      "general_documents": manifest["general"]["documents"],
                      "finance_documents": manifest["finance"]["documents"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
