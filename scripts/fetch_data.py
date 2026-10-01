"""Download the pinned external data listed in configs/data_versions.yaml into data/raw/ and verify checksums.

    python scripts/fetch_data.py
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import zipfile
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def clone(repo: str, commit: str, dest: Path) -> None:
    if not dest.exists():
        subprocess.run(["git", "clone", "-q", repo, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "-q", commit], check=True)


def main() -> int:
    v = yaml.safe_load((ROOT / "configs" / "data_versions.yaml").read_text())
    RAW.mkdir(parents=True, exist_ok=True)
    clone(v["mofgalaxynet"]["repo"], v["mofgalaxynet"]["commit"], RAW / "MOFGalaxyNet")

    q = v["qmof"]
    zpath = RAW / "qmof" / q["file"]
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists() or md5(zpath) != q["md5"]:
        print(f"downloading {q['url']} ({q['size_bytes'] / 1e6:.0f} MB)")
        with requests.get(q["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with zpath.open("wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
    if md5(zpath) != q["md5"]:
        print("QMOF checksum mismatch")
        return 1
    with zipfile.ZipFile(zpath) as z:
        for name in q["extracted"]:
            z.extract(name, zpath.parent)
            if md5(zpath.parent / name) != q["extracted"][name]["md5"]:
                print(f"checksum mismatch for {name}")
                return 1
    print("data OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
