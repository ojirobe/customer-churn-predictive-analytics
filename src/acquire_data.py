from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "Telco-Customer-Churn.csv"
METADATA_PATH = RAW_DIR / "acquisition_metadata.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "--fail", "--location", "--retry", "3", "--output", str(DATA_PATH), SOURCE_URL],
        check=True,
    )
    payload = DATA_PATH.read_bytes()
    metadata = {
        "source_url": SOURCE_URL,
        "publisher": "IBM",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_name": DATA_PATH.name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

