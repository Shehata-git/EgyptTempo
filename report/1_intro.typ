= Introduction & Project Objective

This project implements a complete end-to-end Big Data pipeline for spatio-temporal climate
prediction over Egypt. Using NASA MODIS MOD11C1 land surface temperature observations from
2014–2024, processed through a MongoDB-backed ingestion pipeline and a 4-stage ETL workflow,
the project trains a custom `SpatioTemporalConvLSTM` neural network to forecast Egypt's daily
spatial temperature grid one day ahead.

== Project Objectives

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, left),
    table.header([*Objective*], [*How Satisfied*]),
    [Acquire large-scale real geospatial climate data],
      [NASA MODIS MOD11C1 via `earthaccess` API, 2014–2024, ≈3 650 HDF4 granules],
    [Demonstrate a Big Data ingestion workflow],
      [HDF4 → Egypt crop → scale/impute → MongoDB upsert (idempotent)],
    [Apply MapReduce-style analytics],
      [MongoDB Aggregation Pipeline: temporal avg LST & spatial max LST heatmap],
    [Engineer ML-ready features under RAM constraints],
      [14-day sliding-window ConvLSTM tensors; yearly `.tar.gz` archives],
    [Train a spatio-temporal deep learning model],
      [`SpatioTemporalConvLSTM` — 22 epochs, Kaggle P100; best val loss 0.0081],
  ),
  caption: [Project Objectives]
)

== Why This Is a Big Data Problem

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Dimension*], [*Evidence in This Project*]),
    [Volume],
      [10 years × ~365 granules × ~50 000 valid Egypt pixels ≈ *180 M+ MongoDB documents*; each year's scaled array is (D, 200, 260) float32],
    [Velocity],
      [Daily temporal granularity; year-at-a-time pipeline to fit within Colab/Kaggle 12–16 GB memory],
    [Variety],
      [Source: NASA HDF4 satellite imagery; storage: NoSQL MongoDB; output: NumPy binary tensors → PyTorch DataLoader],
    [Veracity],
      [Fill-value masking (DN=0 → NaN), scale factor ×0.02, NaN imputation with daily spatial mean, upsert deduplication, NaN-to-num guard in train loop],
  ),
  caption: [Big Data Dimensions]
)

== Technology Stack

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Layer*], [*Technology*]),
    [Data Source], [NASA EarthData — MOD11C1 v006 (MODIS Terra, HDF4)],
    [Ingestion API], [`earthaccess` (environment token strategy)],
    [Database], [MongoDB 7.0 (local to Colab/Kaggle VM)],
    [ETL Runtime], [Python 3.14 via `uv`; `pyhdf`, `pymongo`, `numpy`, `scikit-learn`],
    [Analytics], [MongoDB Aggregation Pipeline (MapReduce pattern)],
    [DL Framework], [PyTorch 2.x — `torch.amp`, `GradScaler`, `DataLoader`],
    [Compute], [Kaggle Notebook — P100 GPU (16 GB VRAM)],
    [Data Transfer], [Google Drive → Kaggle via `tar -xzf` extraction],
  ),
  caption: [Technology Stack]
)

#line(length: 100%)
