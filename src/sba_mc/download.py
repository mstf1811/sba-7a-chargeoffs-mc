"""Download the SBA 7(a) FOIA loan-level files from data.sba.gov.

Source page : https://data.sba.gov/dataset/7a-504-foia
Licence     : U.S. Government Works (public domain in the U.S.)

Usage:  python -m sba_mc.download            # default decades FY2000-FY2019 + FY2020-present
        python -m sba_mc.download --all      # also FY1991-FY1999
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE_URL = "https://data.sba.gov/sites/default/files/uploaded_resources/"
SNAPSHOT = "260930"  # "as of" date encoded in the file names (2026-09-30)

FILES_DEFAULT = [
    "7a_504_foia_data_dictionary.xlsx",
    f"FOIA_7a_FY2000_FY2009_asof_{SNAPSHOT}.csv",
    f"FOIA_7a_FY2010_FY2019_asof_{SNAPSHOT}.csv",
    f"FOIA_7a_FY2020_Present_asof_{SNAPSHOT}.csv",
]
FILES_EXTRA = [f"FOIA_7a_FY1991_FY1999_asof_{SNAPSHOT}.csv"]

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(files: list[str], raw_dir: Path = RAW_DIR, force: bool = False) -> dict:
    """Stream each file to ``raw_dir``; skip files already present with the right size.

    Writes ``raw_dir/manifest.json`` with URL, size, sha256 and download time,
    which the data sheet cites as provenance.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    with requests.Session() as s:
        for name in files:
            url = BASE_URL + name
            dest = raw_dir / name
            remote_size = int(s.head(url, timeout=60).headers.get("Content-Length", -1))
            if dest.exists() and dest.stat().st_size == remote_size and not force:
                print(f"[skip] {name} already downloaded ({remote_size / 1e6:.1f} MB)")
                continue
            print(f"[get ] {name} ({remote_size / 1e6:.1f} MB) ...", flush=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with s.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk_size=1 << 20):
                        f.write(block)
            tmp.replace(dest)
            manifest[name] = {
                "url": url,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "downloaded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            manifest_path.write_text(json.dumps(manifest, indent=2))
            print(f"[done] {name}")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="also download FY1991-FY1999")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    a = p.parse_args()
    download(FILES_DEFAULT + (FILES_EXTRA if a.all else []), force=a.force)


if __name__ == "__main__":
    main()
