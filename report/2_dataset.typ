= Dataset & Preprocessing

== NASA MODIS MOD11C1 — Data Source

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Property*], [*Value*]),
    [Product], [MOD11C1 v006 — Daily LST & Emissivity, CMG 0.05°],
    [Satellite], [Terra (MODIS)],
    [Temporal coverage], [2014-01-01 to 2024-12-31 (10 years, ≈3 650 granules)],
    [Global grid], [3 600 rows × 7 200 cols],
    [Spatial resolution], [0.05° (~5.5 km at equator)],
    [Band used], [`LST_Day_CMG` (daytime land surface temperature)],
    [Native units], [Digital Number (DN); Kelvin = DN × 0.02],
    [Fill value], [DN = 0 (cloud/invalid pixel) → masked as NaN],
    [Egypt crop (rows)], [1 160 – 1 360 (≈ 32°N to 22°N)],
    [Egypt crop (cols)], [4 080 – 4 340 (≈ 24°E to 37°E)],
    [Egypt grid size], [200 rows × 260 cols],
  ),
  caption: [MOD11C1 Dataset Properties]
)

== Stage 0 — EarthData Ingestion to MongoDB

Each daily HDF4 file is downloaded, cropped to Egypt, scaled, imputed, and upserted into MongoDB:

+ *Download:* `earthaccess.search_data(short_name="MOD11C1", bounding_box=(24.0,22.0,37.0,32.0), temporal=(start,end))` then `earthaccess.download(results, "./climate_data")`.
+ *Parse:* `pyhdf.SD(filepath).select('LST_Day_CMG')[1160:1360, 4080:4340]` — extracts the 200×260 Egypt sub-grid.
+ *Clean:*
  - Mask: `grid[grid == 0] = NaN` (remove cloud/fill pixels)
  - Scale: `grid = grid × 0.02` (DN → Kelvin)
  - Impute: `NaN → nanmean(grid)` (replace missing with daily spatial average)
+ *Store:* `collection.update_one({"date": date_str}, {"$set": doc}, upsert=True)` — idempotent, safe to re-run.
+ *Cleanup:* HDF file deleted immediately; `gc.collect()` called to free memory.

MongoDB document schema (collection: `earthaccess_db.temperature_data`):

```
{ "date": "A<YYYY><DDD>",  "temperature_grid": [[...200×260 floats in K...]] }
```

== Stage 1 — Min-Max Scaling

Year grids are fetched from MongoDB and normalised globally:

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),
    [Query pattern], [`{"date": {"$regex": "^A<YEAR>"}}`],
    [Normalisation], [`scaled = (raw_K − 261.54) / 82.42`],
    [global_min], [261.54 K],
    [global_max], [343.96 K],
    [Scaler artifact], [`scaler_params.npy` → `[261.54, 343.96]`],
    [Scaled array], [`raw_scaled_{year}.npy`, shape (D, 200, 260) float32],
  ),
  caption: [Scaling Parameters (from `scaler_params.npy`)]
)

== Stage 2 — Sliding-Window Tensor Construction

#figure(
  table(
    columns: (auto, 1fr),
    align: (left, left),
    table.header([*Parameter*], [*Value*]),
    [Lookback window], [14 days],
    [Forecast horizon], [1 day (t+1)],
    [Channel dimension], [1 (LST only)],
    [X tensor shape (on-disk)], [(N, 14, 200, 260, 1) float32],
    [y tensor shape (on-disk)], [(N, 1, 200, 260, 1) float32],
    [N per year], [D − 14 (≈ 351 for a 365-day year)],
    [Total samples (10 yr)], [≈ 3 490],
  ),
  caption: [Sliding Window Tensor Shapes]
)

== MapReduce Analytics (Aggregation Pipeline)

Two MongoDB aggregation jobs were executed on the January 2023 dataset (~507 587 documents):

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    table.header([*Job*], [*Pipeline*], [*Output*]),
    [Temporal avg],
      [`$group: {_id: "$date", avg: {$avg: "$temperature_k"}}`],
      [Time-series line chart — daily avg LST (°C) over Jan 2023],
    [Spatial max],
      [`$group: {_id: {row,col}, max: {$max: "$temperature_k"}}`],
      [Heatmap — peak LST per pixel; reveals Saharan hot spots vs Nile Delta cool strip],
  ),
  caption: [MongoDB Aggregation Pipeline Jobs]
)

#line(length: 100%)
