"""Upload an already-verified Colab handoff directly to a PRIVATE Kaggle dataset.

In the existing 04b Colab kernel, run after loading this module:

    upload_existing_bundle(TRANSFER_ZIP, export_report)

The token is supplied by the owner via hidden input, never printed or written
to files. This requires network transfer and real Kaggle credentials; neither
is simulated as completed by local tests.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _execute(command, *, runner, env=None):
    """Discard both streams: CLI failures may contain sensitive details."""
    try:
        result = runner(command, env=env, stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise RuntimeError("Kaggle command failed; check your API token, account and connection")
    except OSError:
        raise RuntimeError("Unable to start Kaggle CLI; verify installation and connection") from None


def upload_existing_bundle(zip_path, export_report, *, owner="theotheo66",
                           token_reader=None, runner=None):
    """Verify the prior 04b artifact then submit it to a new private dataset.

    A new random dataset slug is used on each attempt. Existing datasets,
    the source ZIP, user Drive and the reviewed training data are never
    overwritten. The API token is supplied only via the CLI child environment.
    The Kaggle service's processing and visibility must be verified by the user.
    """
    source = Path(zip_path).resolve()
    if not isinstance(export_report, dict):
        raise ValueError("The 04b export_report is missing; use the same active Colab session")
    reported_output = export_report.get("output")
    expected = export_report.get("sha256")
    expected_manifest = export_report.get("manifest_sha256")
    if (not isinstance(reported_output, str) or Path(reported_output).resolve() != source
            or source.name != "130u_kaggle_transfer.zip" or not source.is_file()
            or not _digest(expected) or not _digest(expected_manifest)):
        raise ValueError("04b ZIP or its recorded report is missing or invalid; keep existing files")
    if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{1,63}", owner):
        raise ValueError("Invalid Kaggle account name")
    print("Validating the existing transfer package; no repackaging is needed.", flush=True)
    if _sha_file(source) != expected:
        raise ValueError("Transfer ZIP hash differs from 04b output; upload cancelled")
    with zipfile.ZipFile(source) as archive:
        if archive.namelist().count("transfer_manifest.json") != 1:
            raise ValueError("Transfer ZIP lacks a unique manifest")
        info = archive.getinfo("transfer_manifest.json")
        if info.file_size > 8 * 1024 * 1024:
            raise ValueError("Transfer manifest is unexpectedly large")
        raw = archive.read(info)
        if hashlib.sha256(raw).hexdigest() != expected_manifest:
            raise ValueError("Transfer manifest hash differs from 04b output")
        if json.loads(raw).get("format") != "130u-kaggle-transfer-v1":
            raise ValueError("Unexpected transfer manifest format")

    runner = runner or subprocess.run
    token_reader = token_reader or getpass.getpass
    print("Preparing the official Kaggle command-line client.", flush=True)
    _execute([sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "kaggle"], runner=runner)
    slug = "130u-finllm-private-" + uuid.uuid4().hex[:12]
    dataset_id = f"{owner}/{slug}"
    metadata = {"title": "130U FinLLM private transfer " + slug[-12:], "id": dataset_id,
                "licenses": [{"name": "unknown"}],
                "description": "Private transfer for personal research. Source publication rights may vary."}
    # The temporary directory and ZIP are on the same Colab filesystem in the
    # standard workflow, so an inexpensive hard link avoids copying 1.54 GiB.
    with tempfile.TemporaryDirectory(prefix="130u-kaggle-upload-", dir=source.parent) as directory:
        folder = Path(directory)
        staged = folder / source.name
        try:
            os.link(source, staged)
        except OSError:
            if shutil.disk_usage(folder).free < source.stat().st_size + 64 * 1024 * 1024:
                raise OSError("Not enough temporary disk to stage the transfer ZIP") from None
            shutil.copyfile(source, staged)
        (folder / "dataset-metadata.json").write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")
        token = token_reader("Paste your Kaggle API token (hidden input; do not share it in chat): ").strip()
        if not token or "\n" in token or "\r" in token or "\x00" in token:
            raise ValueError("A valid single-line Kaggle API token is required")
        environment = os.environ.copy()
        environment["KAGGLE_API_TOKEN"] = token
        print("Uploading directly from Colab to a private Kaggle dataset; keep the session connected.", flush=True)
        done = threading.Event()
        started = time.monotonic()

        def heartbeat():
            while not done.wait(20):
                seconds = int(time.monotonic() - started)
                print(f"Upload is still running ({seconds} seconds elapsed; percentage unavailable).", flush=True)

        notifier = threading.Thread(target=heartbeat, daemon=True)
        notifier.start()
        try:
            # Kaggle's official CLI defaults to PRIVATE; never add --public.
            _execute(["kaggle", "datasets", "create", "--keep-tabular", "--path", str(folder)],
                     runner=runner, env=environment)
        finally:
            done.set()
            notifier.join(timeout=1)
            environment.pop("KAGGLE_API_TOKEN", None)
            token = None
    result = {"status": "upload_submitted", "dataset_url": f"https://www.kaggle.com/datasets/{dataset_id}",
              "dataset_id": dataset_id, "requested_private": True,
              "transfer_zip_sha256": expected, "transfer_manifest_sha256": expected_manifest,
              "note": "Open Kaggle to verify processing has finished and visibility is Private before attaching."}
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result
