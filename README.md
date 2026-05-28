# neuralkit

Neural network framework from scratch — NumPy only. No PyTorch, no TensorFlow.

Implements forward pass, backpropagation, and two optimizers by hand.

## What's built

- **Layers:** `Dense` with He initialisation
- **Activations:** `ReLU`, `Sigmoid`, `Softmax`
- **Loss:** `CrossEntropyLoss` (softmax + log-loss fused for stability), `MSELoss`
- **Optimizers:** `SGD` with momentum, `Adam` with weight decay
- **Training loop:** mini-batch shuffle, accuracy tracking, progress bar

## Run

```bash
python3 train.py
```

Trains a 3-layer MLP on a 3-class spiral dataset (900 samples).
Reaches ~99% validation accuracy in 100 epochs.

```
epoch    1/100  loss=0.8739  val_acc=0.5222  [██████████░░░░░░░░░░]
epoch   10/100  loss=0.1256  val_acc=0.9778  [███████████████████░]
epoch  100/100  loss=0.0114  val_acc=0.9944  [███████████████████░]

final val accuracy: 0.9944  (99.4%)
```

## Use in your own code

```python
from nn import Sequential, Dense, ReLU
from optim import Adam
from train import train

model = Sequential(
    Dense(input_dim, 128), ReLU(),
    Dense(128, 64),        ReLU(),
    Dense(64, num_classes),
)

train(model, X_train, y_train, X_val, y_val, epochs=200, lr=1e-3)
```
