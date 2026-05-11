= Results & Evaluation

== Training History

The model was trained for 22 epochs on a Kaggle P100 GPU. Loss values are Huber (SmoothL1)
in Min-Max scaled space. Each epoch processed 1 396 batches (effective batch size = 32 via
gradient accumulation over 16 steps at batch_size=2), taking approximately 13–14 minutes per epoch.

*Table: Epochs 1–11*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right, right, right, right, right),
    table.header([*Ep.*], [*1*], [*2*], [*3*], [*4*], [*5*], [*6*], [*7*], [*8*], [*9*], [*10*], [*11*]),
    [Train], [0.1936], [0.1366], [0.0605], [0.0160], [0.0066], [0.0062], [0.0057], [0.0055], [0.0055], [0.0053], [0.0054],
    [Val],   [0.1084], [0.0502], [*0.0081*], [0.0157], [0.0197], [0.0241], [0.0236], [0.0179], [0.0262], [0.0333], [0.0372],
  ),
  caption: [Training & Validation Huber Loss — Epochs 1–11 (best val highlighted)]
)

*Table: Epochs 12–22*

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right, right, right, right, right, right),
    table.header([*Ep.*], [*12*], [*13*], [*14*], [*15*], [*16*], [*17*], [*18*], [*19*], [*20*], [*21*], [*22*]),
    [Train], [0.0054], [0.0052], [0.0053], [0.0053], [0.0053], [0.0054], [0.0054], [0.0054], [0.0053], [0.0053], [0.0053],
    [Val],   [0.0269], [0.0217], [0.0243], [0.0152], [0.0357], [0.0521], [0.0444], [0.0383], [0.0160], [0.0138], [0.0375],
  ),
  caption: [Training & Validation Huber Loss — Epochs 12–22]
)

== Loss Curve

#figure(
  image("visuals/training_history.png", width: 90%),
  caption: [
    Model Training History (Epochs 1–22). Blue = Train Loss, Red = Validation Loss.
    Best validation loss of 0.0081 achieved at Epoch 3. Training converges to a plateau
    near 0.0053 from Epoch 5 onward, while validation oscillates due to the random 80/20 split.
  ]
)

=== Loss Curve Interpretation

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    table.header([*Phase*], [*Observation*], [*Conclusion*]),
    [Epochs 1–3],
      [Both train and val loss descend steeply in parallel],
      [Model actively learns Egypt's spatial LST structure; generalisation is strong],
    [Epoch 3],
      [Val loss hits minimum: *0.0081* while train = 0.0605],
      [Best generalisation point — equivalent to ~7.4 K RMSE in temperature space],
    [Epochs 5–22],
      [Train plateaus at 0.0053; val oscillates 0.013–0.052],
      [Moderate overfitting from random temporal split; AdamW + grad clipping contain divergence],
  ),
  caption: [Loss Curve Phase Analysis]
)

== Spatial Prediction vs Ground Truth

The model was evaluated on a sample from the held-out data (Day 15 of the test period).
Predictions were inverse-transformed from the Min-Max scaled space back to Kelvin and then
converted to Celsius: `°C = (scaled × 82.42 + 261.54) − 273.15`.

#figure(
  image("visuals/prediction_vs_actual.png", width: 100%),
  caption: [
    Real Temperature Forecast — Day 15 of test set.
    *Left:* Actual NASA MODIS LST (°C). *Right:* Model prediction (°C).
    MAE = 20.35 °C on this sample.
  ]
)

=== Spatial Prediction Analysis

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, left),
    table.header([*Observation*], [*Interpretation*]),
    [Model reproduces broad spatial structure: cooler north, hotter south],
      [ConvLSTM successfully learned Egypt's geographic temperature gradient (Mediterranean coast vs Sahara)],
    [Nile River corridor visible as a cooler band in both maps],
      [The 14-day lookback captures the persistent thermal signature of the Nile Valley],
    [Prediction over-estimates temperature across most of Egypt (~20 °C MAE)],
      [Likely caused by early-stopped generalisation (best weights at ep.3 not saved) and random-split leakage; model predicts "warm-biased" spatial patterns],
    [Color scale range differs (actual: 10–60°C, predicted: ~40–65°C)],
      [Systematic positive bias — model learned the average warm season spatial pattern more strongly than cold anomalies (winter/night cloud-free pixels)],
  ),
  caption: [Spatial Prediction Map Analysis]
)

== Summary Metrics

#figure(
  table(
    columns: (1fr, auto),
    align: (left, right),
    table.header([*Metric*], [*Value*]),
    [Best validation loss (Huber, scaled space)], [*0.0081* at Epoch 3],
    [Final training loss (Epoch 22)], [0.0053],
    [Final validation loss (Epoch 22)], [0.0375],
    [Train–Val gap at final epoch], [0.0322],
    [MAE on test sample (Day 15, °C)], [*20.35 °C*],
    [Approx. RMSE at best val#footnote[√0.0081 × 82.42 K ≈ temperature RMSE estimate]],
      [≈ 7.4 K],
    [Epochs to training plateau], [~5],
    [Total epochs trained], [22],
    [Checkpoint size (`climate_convlstm_best.pth`)], [1.79 MB],
  ),
  caption: [Summary Evaluation Metrics]
)

== Discussion

=== What Worked

- *Mixed precision + gradient accumulation:* Enabled training on 200×260 spatial grids with batch_size=2 (effective 32) within 16 GB VRAM, at ≈1.70 it/s per epoch.
- *Memory-mapped DataLoader:* Served all 3 490 samples from 10 yearly `.npy` files with no RAM overflow.
- *Training converged:* Loss reduced ×36 from epoch 1 (0.1936) to plateau (0.0053).
- *Spatial structure captured:* The model reproduces Egypt's north–south temperature gradient and the Nile corridor signature — demonstrating genuine spatial learning.

=== Limitations & Future Work

#figure(
  table(
    columns: (1fr, 1fr),
    align: (left, left),
    table.header([*Limitation*], [*Proposed Fix*]),
    [Random 80/20 split leaks temporal info across years], [Strict year-level split: train 2014–2020, val 2021–2022, test 2023–2024],
    [Best weights (ep. 3) not automatically checkpointed], [Save `state_dict()` whenever `val_loss < best_val_loss`],
    [20.35 °C MAE on test — warm bias], [Calibrate using val set bias correction; add nighttime LST channel],
    [Single channel (daytime LST only)], [Add `LST_Night_CMG` or land-cover auxiliary bands],
    [No quantitative test-set sweep], [Run all 2024 samples through best checkpoint; report full MAE/RMSE distribution],
  ),
  caption: [Limitations and Future Work]
)

#line(length: 100%)
