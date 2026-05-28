import numpy as np
from nn import Sequential, Dense, ReLU, CrossEntropyLoss
from optim import Adam


def accuracy(model, X, y):
    logits = model(X)
    return (logits.argmax(axis=1) == y).mean()


def train(model, X_train, y_train, X_val, y_val,
          epochs=100, batch_size=64, lr=1e-3, verbose=True):

    loss_fn   = CrossEntropyLoss()
    optimizer = Adam(lr=lr)

    N = X_train.shape[0]
    history = {'loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        # Shuffle
        idx = np.random.permutation(N)
        X_s, y_s = X_train[idx], y_train[idx]

        epoch_loss = 0.0
        steps      = 0

        for start in range(0, N, batch_size):
            xb = X_s[start:start + batch_size]
            yb = y_s[start:start + batch_size]

            logits        = model(xb)
            loss, grad    = loss_fn(logits, yb)
            epoch_loss   += loss
            steps        += 1

            optimizer.zero_grad(model.parameters())
            model.backward(grad)
            optimizer.step(model.parameters())

        avg_loss = epoch_loss / steps
        val_acc  = accuracy(model, X_val, y_val)
        history['loss'].append(avg_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            bar   = '█' * int(val_acc * 20) + '░' * (20 - int(val_acc * 20))
            print(f'  epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}  [{bar}]')

    return history


# ── Spiral dataset ────────────────────────────────────────────────────────────

def make_spiral(N=300, classes=3, noise=0.15):
    X, y = [], []
    for c in range(classes):
        ix   = np.arange(N)
        r    = ix / N
        t    = ix / N * 4 + c * (4 / classes) + np.random.randn(N) * noise
        X.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
        y.append(np.full(N, c))
    return np.vstack(X), np.hstack(y)


if __name__ == '__main__':
    np.random.seed(42)

    X, y  = make_spiral(N=300, classes=3)
    idx   = np.random.permutation(len(X))
    X, y  = X[idx], y[idx]           # shuffle before split — spiral is class-ordered
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    model = Sequential(
        Dense(2, 64), ReLU(),
        Dense(64, 64), ReLU(),
        Dense(64, 3),
    )

    print('\n  neuralkit — training on spiral (3-class, 900 samples)\n')
    history = train(model, X_train, y_train, X_val, y_val,
                    epochs=100, batch_size=32, lr=3e-3)

    final_acc = accuracy(model, X_val, y_val)
    print(f'\n  final val accuracy: {final_acc:.4f}  ({final_acc*100:.1f}%)\n')
