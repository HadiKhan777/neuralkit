# neuralkit

[hadikhan777.github.io/portfolio](https://hadikhan777.github.io/portfolio/)

Neural network framework from scratch — NumPy only. No PyTorch, no TensorFlow, no sklearn.

Every forward pass, backward pass, optimizer step, and layer norm is written by hand from the mathematical definitions.

## Features

**Layers**
- `Dense` — fully connected with He initialisation
- `BatchNorm1d` — with running stats for eval mode (full backward derivation, not approximate)
- `LayerNorm` — normalises across the feature dimension per sample
- `Dropout` — inverted dropout, no inference-time change needed
- `Flatten`

**Activations**
- `ReLU`, `LeakyReLU`, `Sigmoid`, `Tanh`
- `GELU` — tanh approximation used in GPT/BERT

**Optimisers**
- `SGD` — with momentum and Nesterov
- `Adam` — adaptive learning rates
- `AdamW` — decoupled weight decay (Loshchilov & Hutter, 2019)

**LR Schedulers**
- `StepLR`, `CosineAnnealingLR`, `WarmupCosineScheduler`

**Losses**
- `CrossEntropyLoss` — fused softmax + CE for numerical stability
- `MSELoss`, `BCELoss`

**Datasets** (`data.py`)
- `make_spiral`, `make_moons`, `make_circles`, `make_xor`, `make_regression`
- `train_val_split`, `normalize`

**Metrics** (`metrics.py`)
- Accuracy, precision, recall, F1
- Confusion matrix with coloured ASCII output
- ASCII training curves (loss + val_acc sparklines)

**Model utilities**
- `model.train()` / `model.eval()` — propagates to BatchNorm and Dropout
- `model.save(path)` / `model.load(path)` — weights via `np.savez`
- `model.summary()` — layer list with parameter counts

## Quick start

```bash
# Default: spiral dataset, 150 epochs
python train.py

# Moons dataset, deep architecture, 200 epochs
python train.py --dataset moons --arch deep --epochs 200

# XOR with wide network and dropout
python train.py --dataset xor --arch wide --dropout 0.2

# Save weights after training
python train.py --dataset circles --save model
```

**Architectures** (`--arch`): `default` · `deep` · `wide` · `minimal`  
**Datasets** (`--dataset`): `spiral` · `moons` · `circles` · `xor`

## Example output

```
  neuralkit — moons / default / 80 epochs

  Layer                    Params
  ────────────────────── ────────
  Dense(2, 64)                192
  BatchNorm1d(64)             128
  ReLU()                        0
  Dense(64, 64)             4,160
  BatchNorm1d(64)             128
  ReLU()                        0
  Dense(64, 2)                130
  ──────────────────────────────
  Total                     4,738

  epoch    1/80  loss=0.2362  val_acc=0.9250  [██████████████████░░]  lr=3.00e-03
  epoch   40/80  loss=0.0215  val_acc=1.0000  [████████████████████]  lr=1.52e-03
  epoch   80/80  loss=0.0211  val_acc=1.0000  [████████████████████]  lr=3.00e-05

  Accuracy   1.0000  (100.0%)
  Precision  1.0000  (macro avg)
  Recall     1.0000  (macro avg)
  F1         1.0000  (macro avg)

  Confusion Matrix

       0      1
  ──────────────────
  0    54     0
  1    0      66
```

## Why

Understanding how deep learning actually works means re-implementing it: the chain rule unrolled through every layer, why batch norm has two paths in forward/backward, why AdamW decouples weight decay from gradient scaling. This does all of that in ~600 lines of NumPy.
