# Spatio-Temporal Climate Prediction on NASA MODIS LST

> Big Data Practical Task — Faculty of Computer Science & Information Systems, May 2026

A complete end-to-end Big Data pipeline that ingests 10 years of NASA MODIS Land Surface
Temperature (MOD11C1) satellite data over Egypt, processes it through a MongoDB-backed ETL
workflow, and trains a custom **ConvLSTM** model to forecast Egypt's spatial temperature
grid one day ahead.

---

## Environment Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

### Prerequisites

- Python 3.13+ (declared in `.python-version`)
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed

### Install & sync dependencies

```bash
# Install uv if not already present
curl -Ls https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies from uv.lock
uv sync
```

### Run any script

```bash
uv run python data-etl/scripts/0_earthaccess_to_mongo.py --year 2023
```

### Required environment variables

| Variable         | Purpose                                      |
|------------------|----------------------------------------------|
| `EARTHDATA_TOKEN`| NASA EarthData bearer token (earthaccess auth)|
| `MONGO_URI`      | MongoDB connection string (default: `mongodb://localhost:27017`) |

---

## Project Structure

```
.
├── data-etl/
│   └── scripts/
│       ├── 0_earthaccess_to_mongo.py   # Download HDF4 → crop Egypt → insert MongoDB
│       ├── 1_fetch_and_scale.py        # MongoDB → MinMax scale → raw_scaled_{year}.npy
│       ├── 2_build_tensors.py          # Sliding window (T=14) → X_{year}.npy, y_{year}.npy
│       ├── 3_pack_and_move.py          # tar.gz compress → move to Google Drive
│       ├── 4_cleanup_mongo.py          # Delete year's documents from MongoDB
│       ├── run_pipeline.sh             # Orchestrator: runs stages 0–4 for each year
│       ├── consumer_dataloader.py      # PyTorch Dataset w/ memory-mapped lazy loading
│       └── consumer_ingestion.md       # DL team guide: extract tarballs from Drive
│
├── notebooks/
│   ├── 00_environment_setup.ipynb          # Colab/Kaggle env validation & dependency check
│   ├── 01_earthaccess_ingestion_and_eda.ipynb  # EarthAccess ingestion + MongoDB MapReduce EDA
│   ├── 02_etl_pipeline_dev.ipynb           # ETL pipeline development & verification
│   └── 03_model_training_and_evaluation.ipynb  # ConvLSTM training (22 epochs) + evaluation plots
│
├── report/
│   ├── main.typ                        # Typst report entry point
│   ├── 0_frontmatter.typ … 5_evaluation.typ
│   ├── visuals/                        # Extracted notebook plots (loss curve, prediction map)
│   ├── assets/                         # University logos
│   └── sources/                        # Markdown source files for each report section
│
├── misc/                               # Planning docs, modeling spec, practical notebooks
├── pyproject.toml                      # Project metadata & dependencies
├── .python-version                     # Pinned Python version (3.13)
└── uv.lock                             # Locked dependency tree
```

---

## Pipeline Overview

```
NASA EarthData (HDF4)
    │
    ▼  Stage 0: earthaccess → pyhdf → Egypt crop → MongoDB upsert
    │
    ▼  Stage 1: MongoDB fetch → MinMax scale [261.54 K, 343.96 K] → .npy
    │
    ▼  Stage 2: Sliding window T=14 → X (N,14,200,260,1) + y (N,1,200,260,1)
    │
    ▼  Stage 3: tar.gz → Google Drive
    │
    ▼  Stage 4: MongoDB cleanup
    │
    ▼  Kaggle: extract tarballs → TrainNumpyDataset (mmap) → ConvLSTM training
```

Run all stages for a year range:

```bash
START_YEAR=2014 END_YEAR=2024 bash data-etl/scripts/run_pipeline.sh
```

---

## Model

**`SpatioTemporalConvLSTM`** — 2-layer stacked ConvLSTM with BatchNorm, collapsed by Conv3D:

| Setting          | Value                              |
|------------------|------------------------------------|
| Input shape      | `(batch, 14, 1, 200, 260)`         |
| Output shape     | `(batch, 1, 200, 260)`             |
| Hidden dim       | 64                                 |
| Kernel size      | 3×3                                |
| Loss             | Huber (SmoothL1)                   |
| Optimizer        | AdamW (`lr=1e-4`, `wd=1e-2`)       |
| Training         | 22 epochs, Kaggle P100, mixed fp16 |
| Best val loss    | 0.0081 (Epoch 3)                   |
| Test MAE (Day 15)| 20.35 °C                           |

---

## Data Source

NASA MODIS Terra — **MOD11C1 v006** (Daily LST & Emissivity, CMG 0.05°)  
Retrieved via [`earthaccess`](https://earthaccess.readthedocs.io/) — no dataset download required;
data is fetched directly from NASA EarthData at pipeline runtime.
