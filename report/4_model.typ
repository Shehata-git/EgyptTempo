= Model Architecture

== ConvLSTMCell — Building Block

The fundamental unit is `ConvLSTMCell`, which replaces LSTM's matrix multiplications with 2D
spatial convolutions, allowing the model to directly process 200×260 temperature grids without
flattening them.

A *single fused* `Conv2d` computes all four gates (input, forget, cell, output) simultaneously.
The input to this convolution is the *concatenation* of the current spatial input and the
previous hidden state along the channel dimension.

```python
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        # Fused: computes i, f, g, o gates in one convolution
        self.conv = nn.Conv2d(
            in_channels + out_channels,  # concat(x_t, h_t-1)
            4 * out_channels,            # stacked gate outputs
            kernel_size, padding=padding
        )

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.conv(torch.cat([x, h_prev], dim=1))
        i, f, g, o = torch.split(gates, gates.size(1) // 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c_prev + i * g      # cell state update
        h_next = o * torch.tanh(c_next)  # hidden state update
        return h_next, c_next
```

=== ConvLSTMCell Parameter Count (Layer 1)

For `in_channels=1`, `out_channels=64`, `kernel_size=3`:

#figure(
  table(
    columns: (auto, 1fr, auto),
    align: (left, left, right),
    table.header([*Component*], [*Shape*], [*Parameters*]),
    [`conv.weight`], [`Conv2d(65, 256, 3×3)`], [149 760],
    [`conv.bias`], [`(256,)`], [256],
    [*Cell 1 total*], [], [*150 016*],
  ),
  caption: [ConvLSTMCell Parameter Count (Layer 1: in=1→64)]
)

== SpatioTemporalConvLSTM — Full Model

The full model stacks two ConvLSTM layers (each followed by Batch Normalisation) and collapses
the 14-step hidden state sequence to a single spatial prediction using a Conv3D layer.

```python
class SpatioTemporalConvLSTM(nn.Module):
    def __init__(self, num_channels, hidden_dim, kernel_size, seq_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell1 = ConvLSTMCell(num_channels, hidden_dim,
                                   kernel_size, padding=kernel_size//2)
        self.bn1   = nn.BatchNorm2d(hidden_dim)
        self.cell2 = ConvLSTMCell(hidden_dim, hidden_dim,
                                   kernel_size, padding=kernel_size//2)
        self.bn2   = nn.BatchNorm2d(hidden_dim)
        # Collapse 14-step output sequence → 1-step prediction
        self.conv_final = nn.Conv3d(hidden_dim, num_channels,
                                     kernel_size=(seq_len, 1, 1), padding=0)

    def forward(self, x):
        b, t, c, h, w = x.size()  # (batch, 14, 1, 200, 260)
        h1 = c1 = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        h2 = c2 = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        outputs = []
        for step in range(t):
            h1, c1 = self.cell1(x[:, step], (h1, c1))
            h1 = self.bn1(h1)
            h2, c2 = self.cell2(h1, (h2, c2))
            h2 = self.bn2(h2)
            outputs.append(h2.unsqueeze(2))          # (B, 64, 1, H, W)
        output_seq = torch.cat(outputs, dim=2)        # (B, 64, 14, H, W)
        prediction = self.conv_final(output_seq)      # (B, 1, 1, H, W)
        return prediction.squeeze(2)                  # (B, 1, H, W)
```

=== Layer Summary

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, left, left, left),
    table.header([*Layer*], [*Type*], [*Input Shape*], [*Output Shape*]),
    [`cell1`], [ConvLSTMCell(1→64)], [(B, 1, 200, 260)], [(B, 64, 200, 260)],
    [`bn1`],   [BatchNorm2d(64)],    [(B, 64, 200, 260)], [(B, 64, 200, 260)],
    [`cell2`], [ConvLSTMCell(64→64)], [(B, 64, 200, 260)], [(B, 64, 200, 260)],
    [`bn2`],   [BatchNorm2d(64)],    [(B, 64, 200, 260)], [(B, 64, 200, 260)],
    [`conv_final`], [Conv3d(64→1, k=(14,1,1))], [(B, 64, 14, 200, 260)], [(B, 1, 1, 200, 260)],
    [squeeze(2)], [—], [(B, 1, 1, 200, 260)], [(B, 1, 200, 260)],
  ),
  caption: [SpatioTemporalConvLSTM Layer-by-Layer Summary]
)

=== Instantiation

```python
model = SpatioTemporalConvLSTM(
    num_channels=1,   # single LST channel
    hidden_dim=64,    # hidden state depth
    kernel_size=3,    # 3×3 spatial receptive field
    seq_len=14        # matches 14-day lookback
).to(device)
```

== Training Configuration

=== Loss, Optimizer, Scheduler

#figure(
  table(
    columns: (auto, auto, 1fr),
    align: (left, left, left),
    table.header([*Setting*], [*Value*], [*Rationale*]),
    [Loss], [`nn.SmoothL1Loss()` (Huber)], [Robust to extreme LST outliers (cloud contamination)],
    [Optimizer], [`AdamW(lr=1e-4, weight_decay=1e-2)`], [Adam with explicit L2 regularisation],
    [LR scheduler], [`ReduceLROnPlateau(patience=5)`], [Auto-halves LR when val loss stagnates],
    [Epochs], [22], [Trained to completion on Kaggle P100],
    [Batch size], [2], [Each sample ≈57 MB float32; VRAM-constrained],
    [Gradient accum.], [16 steps], [Effective batch = 32; avoids OOM],
  ),
  caption: [Training Hyperparameters]
)

=== GPU Efficiency Techniques

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: (left, left, left),
    table.header([*Technique*], [*Implementation*], [*Purpose*]),
    [Mixed Precision], [`autocast('cuda')` + `GradScaler`], [fp16 forward pass — ~½ VRAM, ~2× faster],
    [Gradient Accumulation], [Divide loss by 16; step every 16 batches], [Effective batch=32 without extra VRAM],
    [Gradient Clipping], [`clip_grad_norm_(model, max_norm=1.0)`], [Prevents ConvLSTM exploding gradients],
    [NaN Guard], [`torch.nan_to_num(inputs, nan=0.0)`], [Handles residual cloud NaN in tensors],
    [Label Shape Fix], [`labels.squeeze(2)` if dim==5 and size(2)==1], [Fixes (B,1,1,H,W)→(B,1,H,W) mismatch],
    [VRAM Cleanup], [`empty_cache(); gc.collect()` per epoch], [Prevents fragmentation over 22 epochs],
  ),
  caption: [GPU Efficiency Techniques in `train_model()`]
)

#line(length: 100%)
