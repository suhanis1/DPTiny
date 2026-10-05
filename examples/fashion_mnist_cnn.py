"""Train the baseline MNIST CNN on Fashion-MNIST with DPTiny.

Run from the repository root:  python3 examples/fashion_mnist_cnn.py
"""
import time

import numpy as np

from dptiny import Variable, no_grad, softmax_cross_entropy, test_mode
from dptiny.backend import xp
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

SEED = 0
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001


def build_model():
    """Baseline architecture from examples/mnist_cnn.py, unchanged."""
    return Sequential(
        Conv2d(1, 16, 3, pad=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(16, 32, 3, pad=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(32 * 7 * 7, 128),
        ReLU(),
        Dropout(0.3),
        Linear(128, 10),
    )


def predict(model, loader):
    """Predicted labels for every sample of an unshuffled loader."""
    preds = []
    with test_mode(), no_grad():
        for x, _ in loader:
            preds.append(np.asarray(model(Variable(x)).data).argmax(axis=1))
    return np.concatenate(preds)


def confusion_matrix(y_true, y_pred, n_classes=10):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)  # rows = true class, columns = predicted
    return cm


def most_confused_pair(cm):
    """Unordered pair (i, j), i < j, with the most errors in both directions."""
    best, best_count = (0, 1), -1
    for i in range(cm.shape[0]):
        for j in range(i + 1, cm.shape[0]):
            count = int(cm[i, j] + cm[j, i])
            if count > best_count:
                best, best_count = (i, j), count
    return best, best_count


def main():
    np.random.seed(SEED)
    xp.random.seed(SEED)

    print("Loading Fashion-MNIST dataset...")
    X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

    # Standardize with the training-set mean and standard deviation.
    mean = float(X_train.mean())
    std = float(X_train.std())
    X_train = ((X_train - mean) / std).astype(np.float32)
    X_test = ((X_test - mean) / std).astype(np.float32)
    print(f"Training-set mean = {mean:.4f}, std = {std:.4f}")

    model = build_model()
    data_loader = DataLoader((X_train, y_train), BATCH_SIZE)
    test_loader = DataLoader((X_test, y_test), BATCH_SIZE, shuffle=False)
    optimizer = Adam(model, lr=LR)

    start_time = time.time()
    for epoch in range(EPOCHS):
        sum_loss, correct, count = 0.0, 0, 0
        model.train()
        for x, t in data_loader:
            y = model(Variable(x))
            loss = softmax_cross_entropy(y, t)

            model.cleargrads()
            loss.backward()
            optimizer.update()

            sum_loss += float(loss.data) * len(t)
            correct += int((np.asarray(y.data).argmax(axis=1) == t).sum())
            count += len(t)

        test_acc = float((predict(model, test_loader) == y_test).mean())
        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"train loss {sum_loss / count:.4f} | "
            f"train acc {correct / count:.4f} | "
            f"test acc {test_acc:.4f} | "
            f"time {time.time() - start_time:.1f}s"
        )

    # ---------------- Final evaluation ----------------
    y_pred = predict(model, test_loader)
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion matrix (rows = true class, columns = predicted class):")
    print("     " + "".join(f"{k:>6d}" for k in range(10)))
    for k in range(10):
        print(f"{k:>4d} " + "".join(f"{v:>6d}" for v in cm[k]))

    print("\nPer-class accuracy:")
    for k, name in enumerate(CLASSES):
        print(f"  {k} {name:<12s} {cm[k, k] / cm[k].sum():.4f}")

    (i, j), count = most_confused_pair(cm)
    print(
        f"\nMost confused pair: {CLASSES[i]} <-> {CLASSES[j]} "
        f"({count} errors in total: {cm[i, j]} {CLASSES[i]} predicted as "
        f"{CLASSES[j]}, {cm[j, i]} the reverse)"
    )


if __name__ == "__main__":
    main()
