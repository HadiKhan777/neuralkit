"""
neuralkit train — configurable training script.

Usage:
  python train.py
  python train.py --dataset moons --arch deep --epochs 200
  python train.py --dataset circles --arch wide --lr 1e-3 --dropout 0.2
  python train.py --dataset xor --save model
"""

import argparse
import numpy as np

from nn      import Sequential, Dense, ReLU, LeakyReLU, GELU, BatchNorm1d, Dropout, CrossEntropyLoss
from optim   import AdamW, CosineAnnealingLR
from data    import make_spiral, make_moons, make_circles, make_xor, train_val_split, normalize
from metrics import accuracy, confusion_matrix, precision_recall_f1, print_confusion_matrix, plot_history


DATASETS = {
    'spiral':  lambda: make_spiral(N=300, classes=3),
    'moons':   lambda: make_moons(N=600),
    'circles': lambda: make_circles(N=600),
    'xor':     lambda: make_xor(N=600),
}


def build_model(arch, in_features, n_classes, dropout=0.0):
    d = max(0.0, dropout)
    if arch == 'default':
        return Sequential(
            Dense(in_features, 64), BatchNorm1d(64), ReLU(),
            Dense(64, 64),          BatchNorm1d(64), ReLU(),
            Dense(64, n_classes),
        )
    if arch == 'deep':
        return Sequential(
            Dense(in_features, 128), BatchNorm1d(128), GELU(),
            Dense(128, 128),         BatchNorm1d(128), GELU(),
            Dense(128, 64),          BatchNorm1d(64),  GELU(),
            Dense(64, n_classes),
        )
    if arch == 'wide':
        layers = [Dense(in_features, 256), LeakyReLU()]
        if d: layers.append(Dropout(d))
        layers += [Dense(256, 256), LeakyReLU()]
        if d: layers.append(Dropout(d))
        layers.append(Dense(256, n_classes))
        return Sequential(*layers)
    if arch == 'minimal':
        return Sequential(
            Dense(in_features, 32), ReLU(),
            Dense(32, n_classes),
        )
    raise ValueError(f'Unknown arch: {arch}')


def train(model, X_train, y_train, X_val, y_val,
          epochs=150, batch_size=32, lr=3e-3):

    loss_fn   = CrossEntropyLoss()
    optimizer = AdamW(lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    N       = X_train.shape[0]
    history = {'loss': [], 'val_acc': [], 'lr': []}

    log_every = max(1, epochs // 10)

    for epoch in range(1, epochs + 1):
        model.train()
        idx  = np.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss, steps = 0.0, 0
        for start in range(0, N, batch_size):
            xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]
            logits      = model(xb)
            loss, grad  = loss_fn(logits, yb)
            epoch_loss += loss
            steps      += 1
            optimizer.zero_grad(model.parameters())
            model.backward(grad)
            optimizer.step(model.parameters())

        scheduler.step()

        model.eval()
        avg_loss = epoch_loss / steps
        preds    = model(X_val).argmax(axis=1)
        val_acc  = accuracy(preds, y_val)

        history['loss'].append(avg_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(optimizer.lr)

        if epoch % log_every == 0 or epoch == 1:
            bar = '█' * int(val_acc * 20) + '░' * (20 - int(val_acc * 20))
            print(f'  epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  '
                  f'val_acc={val_acc:.4f}  [{bar}]  lr={optimizer.lr:.2e}')

    return history


def main():
    parser = argparse.ArgumentParser(description='neuralkit — train from scratch, NumPy only')
    parser.add_argument('--dataset', default='spiral', choices=list(DATASETS.keys()),
                        help='dataset to train on')
    parser.add_argument('--arch',    default='default',
                        choices=['default', 'deep', 'wide', 'minimal'],
                        help='model architecture')
    parser.add_argument('--epochs',  type=int,   default=150)
    parser.add_argument('--batch',   type=int,   default=32)
    parser.add_argument('--lr',      type=float, default=3e-3)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--save',    type=str,   default=None,
                        help='save model weights to this path (no extension)')
    args = parser.parse_args()

    np.random.seed(42)

    print(f'\n  neuralkit — {args.dataset} / {args.arch} / {args.epochs} epochs\n')

    X, y = DATASETS[args.dataset]()
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2)
    X_train, X_val = normalize(X_train, X_val)

    n_classes = len(np.unique(y))
    model     = build_model(args.arch, X_train.shape[1], n_classes, args.dropout)
    model.summary()

    history = train(model, X_train, y_train, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    model.eval()
    preds = model(X_val).argmax(axis=1)
    acc   = accuracy(preds, y_val)
    prec, rec, f1 = precision_recall_f1(preds, y_val, n_classes)

    print(f'\n  ── Final results ──────────────────────────────────')
    print(f'  Accuracy   {acc:.4f}  ({acc*100:.1f}%)')
    print(f'  Precision  {prec.mean():.4f}  (macro avg)')
    print(f'  Recall     {rec.mean():.4f}  (macro avg)')
    print(f'  F1         {f1.mean():.4f}  (macro avg)')

    cm = confusion_matrix(preds, y_val, n_classes)
    print_confusion_matrix(cm, [str(i) for i in range(n_classes)])
    plot_history(history)

    if args.save:
        model.save(args.save)


if __name__ == '__main__':
    main()
