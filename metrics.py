import numpy as np


def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()


def confusion_matrix(y_pred, y_true, n_classes=None):
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(y_pred, y_true, n_classes=None):
    cm  = confusion_matrix(y_pred, y_true, n_classes)
    tp  = np.diag(cm)
    pre = tp / (cm.sum(axis=0) + 1e-12)
    rec = tp / (cm.sum(axis=1) + 1e-12)
    f1  = 2 * pre * rec / (pre + rec + 1e-12)
    return pre, rec, f1


def print_confusion_matrix(cm, class_names=None):
    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]
    w = max(len(c) for c in class_names) + 1
    header = ' ' * (w + 2) + '   '.join(c.center(4) for c in class_names)
    print(f'\n  Confusion Matrix\n')
    print('  ' + header)
    print('  ' + '─' * (w + 2 + n * 7))
    for i, row in enumerate(cm):
        cells = '   '.join(str(v).center(4) for v in row)
        print(f'  {class_names[i]:<{w+1}} {cells}')
    print()


def plot_history(history, width=56):
    """ASCII sparkline plots for loss and val_acc."""
    chars  = '▁▂▃▄▅▆▇█'
    epochs = len(history['loss'])

    def spark(data):
        mn, mx = min(data), max(data)
        if mx == mn:
            return '─' * width
        window = data[-width:]
        return ''.join(chars[min(7, int((v - mn) / (mx - mn) * 7.999))] for v in window)

    losses   = history['loss']
    val_accs = history['val_acc']

    print(f'\n  Training curves ({epochs} epochs)\n')
    print(f'  loss     min={min(losses):.4f}  final={losses[-1]:.4f}')
    print(f'  {"─"*width}')
    print(f'  {spark(losses)}')
    print(f'\n  val_acc  min={min(val_accs):.4f}  final={val_accs[-1]:.4f}')
    print(f'  {"─"*width}')
    print(f'  {spark(val_accs)}')
    print()
