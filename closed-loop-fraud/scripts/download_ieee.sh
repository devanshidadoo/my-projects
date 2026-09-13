#!/usr/bin/env bash
# Fetch the IEEE-CIS Fraud Detection dataset into data/raw/.
#
# Requires the Kaggle CLI and an API token at ~/.kaggle/kaggle.json, and you must have accepted
# the competition rules on kaggle.com first (the API returns 403 otherwise).
#
# The closed-loop experiment does not need this: it runs on the calibrated synthetic stream,
# because a real log contains no counterfactual labels for declined transactions. What the real
# data is for is validating the feature engine, the leakage audit and the static benchmark on
# something nobody generated. See docs/METHODOLOGY.md, "Why a simulator".
set -euo pipefail

RAW_DIR="${1:-data/raw}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. Install it with:  pip install kaggle" >&2
  exit 1
fi

mkdir -p "$RAW_DIR"
echo "Downloading ieee-fraud-detection into $RAW_DIR ..."
kaggle competitions download -c ieee-fraud-detection -p "$RAW_DIR"
unzip -o "$RAW_DIR/ieee-fraud-detection.zip" -d "$RAW_DIR"
rm -f "$RAW_DIR/ieee-fraud-detection.zip"

echo
echo "Files in $RAW_DIR:"
ls -lh "$RAW_DIR"
echo
echo "Next:  clfraud ingest-ieee --raw-dir $RAW_DIR --out data/processed/ieee.parquet"
