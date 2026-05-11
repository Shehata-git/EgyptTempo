= System Architecture

== End-to-End Pipeline Overview

The project is structured as a *year-at-a-time*, memory-constrained pipeline.
Each year's data is ingested, scaled, windowed, compressed, and moved before the next year begins.
This design keeps peak RAM usage below the 12 GB Colab / 16 GB Kaggle limit throughout.

```
NASA EarthData (HDF4 granules)
        │
        ▼
[Stage 0] 0_earthaccess_to_mongo.py
  ├─ earthaccess.search_data()  →  earthaccess.download()
  ├─ pyhdf → crop Egypt [1160:1360, 4080:4340] → scale ×0.02 → impute NaN
  └─ MongoDB upsert (idempotent) → delete HDF file → gc.collect()
        │
        ▼
[Stage 1] 1_fetch_and_scale.py
  ├─ MongoDB query: {"date": {"$regex": "^A<YEAR>"}} sorted ascending
  ├─ Stack grids → (D, 200, 260) float32
  ├─ MinMaxScaler [261.54 K, 343.96 K] → scaler_params.npy
  └─ raw_scaled_{year}.npy → /content/local_staging/
        │
        ▼
[Stage 2] 2_build_tensors.py
  ├─ Load raw_scaled_{year}.npy
  ├─ Sliding window (T=14): X (N,14,200,260,1) and y (N,1,200,260,1)
  ├─ Save X_{year}.npy, y_{year}.npy → /content/local_staging/
  └─ Delete raw_scaled_{year}.npy → gc.collect()
        │
        ▼
[Stage 3] 3_pack_and_move.py
  ├─ tar -czf climate_tensors_{year}.tar.gz  *.npy
  ├─ shutil.move → Google Drive (MyDrive/Data-sets/BigDataData/)
  └─ rm -rf /content/local_staging/*
        │
        ▼
[Stage 4] 4_cleanup_mongo.py
  └─ collection.delete_many({"date": {"$regex": "^A<YEAR>"}})
        │
        ▼
[Repeat for year + 1]
        │
        ▼
[Consumer: Kaggle Notebook]
  ├─ cp BigDataData/*.tar.gz /kaggle/working/
  ├─ tar -xzvf *.tar.gz  →  X_20XX.npy, y_20XX.npy (for all years)
  └─ TrainNumpyDataset (mmap) → DataLoader → SpatioTemporalConvLSTM
```

== ETL Stage Details

=== Stage 0: EarthData Ingestion

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),
    [Authentication], [`strategy="environment"` → reads `EARTHDATA_TOKEN` env var],
    [API call], [`earthaccess.search_data(short_name="MOD11C1", bounding_box=(24,22,37,32), temporal=(...))` ],
    [HDF4 band], [`LST_Day_CMG`],
    [Egypt crop], [`grid[1160:1360, 4080:4340]`],
    [Scale], [DN × 0.02 → Kelvin],
    [Imputation], [`grid = np.nan_to_num(grid, nan=np.nanmean(grid))`],
    [MongoDB key], [`{"date": "A<YYYY><DDD>"}`],
    [Deduplication], [`update_one(..., upsert=True)` — safe on restart],
    [Disk cleanup], [`os.remove(hdf_path)` immediately after parsing],
  ),
  caption: [Stage 0 Parameters]
)

=== Stage 2: Tensor Construction & Year Boundaries

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    table.header([*Year File*], [*Year*], [*Start*], [*End*]),
    [Index 0], [2014], [0], [350],
    [Index 1], [2015], [351], [701],
    [Index 2], [2016], [702], [1044],
    [Index 3], [2017], [1045], [1395],
    [Index 4], [2018], [1396], [1746],
    [Index 5], [2019], [1747], [2097],
    [Index 6], [2020], [2098], [2449],
    [Index 7], [2021], [2450], [2800],
    [Index 8], [2022], [2801], [3138],
    [Index 9], [2023], [3139], [3489],
  ),
  caption: [Year Boundaries in `TrainNumpyDataset` (Total: 3 490 samples)]
)

2024 data (`X_2024.npy`, `y_2024.npy`) exists on disk but is excluded from boundaries —
reserved as the held-out test year.

== Memory-Mapped DataLoader

The `TrainNumpyDataset` opens all yearly `.npy` files as memory maps at init time
(`mmap_mode='r'`) so only the requested slice is read from disk per `__getitem__` call.

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Feature*], [*Implementation Detail*]),
    [Init], [`np.load(f, mmap_mode='r')` for all 10 X and 10 y files — no RAM allocated],
    [`__len__`], [`boundaries[-1][1] + 1 = 3490` total samples],
    [`__getitem__`], [Bisects `boundaries` to find year file; reads `np.array(mmap[local_idx])` (copies slice to RAM)],
    [Permutation], [`x.permute(0,3,1,2)` → `(T,C,H,W)`; `y.permute(0,3,1,2)` → `(1,C,H,W)`],
    [Train/val split], [`torch.utils.data.random_split(3490, [2792, 698])` (80 / 20)],
    [batch_size], [2 (each sample ≈ 57 MB float32 — VRAM constrained)],
    [num_workers], [2 (parallel disk reads)],
    [pin_memory], [True (faster CPU→GPU DMA)],
  ),
  caption: [DataLoader Implementation Details]
)

#line(length: 100%)
