"""Restore the pinned 2,000-image Open Images V7 evaluation cohort."""

from __future__ import annotations

import argparse
from pathlib import Path

from pixelogue.io import read_jsonl, write_jsonl
from pixelogue.open_images import OpenImagesDownloader, OpenImagesPinnedRecord


def main() -> None:
    """Verify official provenance and pixels before writing evaluation records."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=root / "data/open-images-v7-eval-2000")
    args = parser.parse_args()
    pinned = tuple(
        read_jsonl(
            root / "validation/open_images_v7_eval_2000_manifest.jsonl", OpenImagesPinnedRecord
        )
    )
    sources, rights = OpenImagesDownloader().download_sample(
        "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
        args.destination,
        count=len(pinned),
        seed=20261001,
        pinned=pinned,
    )
    write_jsonl(args.destination / "sources.jsonl", sources)
    write_jsonl(args.destination / "rights.jsonl", rights)
    print(f"Restored {len(sources)} pinned evaluation images")


if __name__ == "__main__":
    main()
