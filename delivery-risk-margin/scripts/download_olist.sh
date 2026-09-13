#!/usr/bin/env bash
# Fetch the Olist Brazilian e-commerce release into a directory (default: data/raw).
#
# Needs a Kaggle API token at ~/.kaggle/kaggle.json (Kaggle > Account > Create New API Token)
# and the kaggle CLI:  pip install kaggle
#
#   ./scripts/download_olist.sh data/raw
#   drisk ingest-olist --raw-dir data/raw
set -euo pipefail

DEST="${1:-data/raw}"
DATASET="olistbr/brazilian-ecommerce"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "kaggle CLI not found. pip install kaggle, then place your token at ~/.kaggle/kaggle.json" >&2
  exit 1
fi

mkdir -p "$DEST"
kaggle datasets download -d "$DATASET" -p "$DEST" --unzip
echo "Downloaded to $DEST:"
ls -1 "$DEST"

cat <<'NOTE'

The nine-table load expects these files:
  olist_orders_dataset.csv          olist_order_items_dataset.csv
  olist_customers_dataset.csv       olist_sellers_dataset.csv
  olist_products_dataset.csv        olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv   olist_geolocation_dataset.csv

`shipping_lanes` and `carriers` have no counterpart in the release and are derived from the
training window; src/deliveryrisk/data/olist.py documents exactly how.
NOTE
