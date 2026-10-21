"""Select the columns we use and freeze a train / hold-out split.

Why freeze the hold-out: every candidate model is compared with the current
champion on exactly the same rows, so the release gate is a fair, repeatable
comparison. The file's SHA-256 hash is logged with every model as evidence.

Usage:
    python training/prepare_data.py --raw data/raw/application_train.csv --out-dir data/prepared
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from scoring.config import ID_COLUMN, RANDOM_SEED, RAW_INPUTS, TARGET


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out-dir", default="data/prepared")
    parser.add_argument("--holdout-size", type=float, default=0.2)
    args = parser.parse_args()

    columns = [ID_COLUMN, TARGET] + RAW_INPUTS
    df = pd.read_csv(args.raw, usecols=columns)

    train, holdout = train_test_split(
        df, test_size=args.holdout_size, stratify=df[TARGET], random_state=RANDOM_SEED
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path, holdout_path = out_dir / "train.parquet", out_dir / "holdout.parquet"
    train.to_parquet(train_path, index=False)
    holdout.to_parquet(holdout_path, index=False)

    manifest = {
        "source_file": str(args.raw),
        "source_sha256": file_sha256(Path(args.raw)),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_sha256": file_sha256(train_path),
        "holdout_sha256": file_sha256(holdout_path),
        "default_rate_train": round(float(train[TARGET].mean()), 5),
        "default_rate_holdout": round(float(holdout[TARGET].mean()), 5),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()