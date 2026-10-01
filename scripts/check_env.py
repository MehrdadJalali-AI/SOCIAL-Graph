"""Print Python/package versions, CPU count and RAM, and test internet access to Figshare and GitHub."""

from __future__ import annotations

import importlib.metadata as md
import os
import platform
import sys

PACKAGES = ["numpy", "scipy", "pandas", "pyarrow", "networkx", "python-igraph", "leidenalg", "rdkit",
            "scikit-learn", "mendeleev", "mordredcommunity", "joblib", "pyyaml", "matplotlib", "seaborn",
            "scikit-posthocs", "tqdm", "pytest", "psutil", "requests", "tabulate", "graphsocial"]
URLS = {"figshare": "https://api.figshare.com/v2/articles/13147324", "github": "https://github.com"}


def main() -> int:
    ok = True
    print(f"python   {sys.version.split()[0]} ({platform.python_implementation()}, {platform.platform()})")
    if sys.version_info[:2] != (3, 13):
        print("  WARNING: target is Python 3.13")
    print(f"cpus     {os.cpu_count()}")
    try:
        import psutil

        print(f"ram      {psutil.virtual_memory().total / 2**30:.1f} GiB "
              f"({psutil.virtual_memory().available / 2**30:.1f} GiB available)")
    except ImportError:
        print("ram      (psutil missing)")
    for p in PACKAGES:
        try:
            print(f"  {p:18s} {md.version(p)}")
        except md.PackageNotFoundError:
            ok = False
            print(f"  {p:18s} MISSING")
    try:
        import requests

        for name, url in URLS.items():
            try:
                r = requests.get(url, timeout=10, stream=True)  # figshare API rejects HEAD
                print(f"net      {name:9s} HTTP {r.status_code}")
            except requests.RequestException as exc:
                print(f"net      {name:9s} FAILED ({exc.__class__.__name__})")
    except ImportError:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
