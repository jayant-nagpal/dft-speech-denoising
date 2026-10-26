"""Download FSDD v1.0.10 and verify the pinned archive checksum."""
from pathlib import Path
import argparse
import hashlib
import urllib.request
import zipfile
from .paths import DATA_DIR

URL = "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/tags/v1.0.10"
SHA256 = "c680098f5d2355ba7cc4bd8b0ca9e0a24c00ebe3b4c552208889a4d28d4fbc2b"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path, default=DATA_DIR)
    args = p.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    archive = args.destination / "fsdd-v1.0.10.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError(
            "Archive checksum differs: inspect source/version before proceeding"
        )
    root = args.destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise RuntimeError("Unsafe archive member")
        z.extractall(root)
    print(root / "free-spoken-digit-dataset-1.0.10" / "recordings")
    print("Dataset: FSDD contributors, v1.0.10; CC BY-SA 4.0")


if __name__ == "__main__":
    main()
