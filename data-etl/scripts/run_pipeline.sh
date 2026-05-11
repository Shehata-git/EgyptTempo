#!/bin/bash

START_YEAR=${1:-2014}
END_YEAR=${2:-2024}

echo "========================================"
echo "   CLIMATE DATA PIPELINE ORCHESTRATOR   "
echo "========================================"
echo "Range: $START_YEAR - $END_YEAR"

mkdir -p /content/local_staging 2>/dev/null || mkdir -p ./local_staging

for Y in $(seq $START_YEAR $END_YEAR); do
    echo ""
    echo ">>> Starting Processing for Year: $Y"

    # Step 0: Download and ingest to Mongo
    echo "[Step 0/3] Downloading NASA Data to local MongoDB..."
    uv run python scripts/0_earthaccess_to_mongo.py --year $Y
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: 0_earthaccess_to_mongo.py failed for year $Y."
        exit 1
    fi

    # Step 1: Fetch from MongoDB and Normalize
    echo "[Step 1/3] Fetching data and applying Min-Max scaling..."
    uv run python scripts/1_fetch_and_scale.py --year $Y
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: 1_fetch_and_scale.py failed for year $Y."
        exit 1
    fi

    # Step 2: Create Sliding Window Tensors
    echo "[Step 2/3] Generating sliding window tensors (lookback=14)..."
    uv run python scripts/2_build_tensors.py --year $Y
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: 2_build_tensors.py failed for year $Y."
        exit 1
    fi

    # Step 3: Archive and Transfer to Google Drive
    echo "[Step 3/3] Archiving tensors and transferring to Google Drive..."
    uv run python scripts/3_pack_and_move.py --year $Y
    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: 3_pack_and_move.py failed for year $Y."
        exit 1
    fi

    # Step 4: Drop the Mongo database for that year using Python
    echo "Cleaning up MongoDB to protect Colab system..."
    uv run python scripts/4_cleanup_mongo.py --year $Y

    echo ">>> Successfully completed processing for year $Y."
done

echo "========================================"
echo "   PIPELINE EXECUTION COMPLETED   "
echo "========================================"
