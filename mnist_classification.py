import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

try:
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
except ImportError:
    # Same standard split (first 60,000 train, last 10,000 test) via OpenML
    from sklearn.datasets import fetch_openml
    d = fetch_openml("mnist_784", version=1, as_frame=False)
    X, y = d.data, d.target.astype(int)
    x_train, y_train, x_test, y_test = X[:60000], y[:60000], X[60000:], y[60000:]

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# ---- Part 1: binary "5 detector" with SGD ----
target = (y_train == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(x_train, target)
print("Predict sample 100:", sgd_clf.predict([x_train[100]]))
y_train_pred = cross_val_predict(sgd_clf, x_train, target, cv=3, n_jobs=-1)
print("Binary confusion matrix (SGD, 3-fold CV):\n", confusion_matrix(target, y_train_pred))
print("SGD Precision:", precision_score(target, y_train_pred))
print("SGD Recall:", recall_score(target, y_train_pred))
print("SGD F1:", f1_score(target, y_train_pred))

# ---- Part 2: Random Forest vs SGD (same 5-detector, 3-fold CV) ----
forest_clf = RandomForestClassifier(random_state=42, n_jobs=-1)
y_rf_pred = cross_val_predict(forest_clf, x_train, target, cv=3)
print("RF Precision:", precision_score(target, y_rf_pred))
print("RF Recall:", recall_score(target, y_rf_pred))
print("RF F1:", f1_score(target, y_rf_pred))

# ---- Part 3: multiclass SGD, raw vs scaled ----
sgd_multi = SGDClassifier(random_state=42)
sgd_multi.fit(x_train, y_train)
y_test_pred = sgd_multi.predict(x_test)
conf_mx = confusion_matrix(y_test, y_test_pred)
print("Multiclass confusion matrix (raw pixels):\n", conf_mx)
print("Accuracy (raw):", accuracy_score(y_test, y_test_pred))

scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train.astype(np.float64))
xs_test = scaler.transform(x_test.astype(np.float64))
sgd_scaled = SGDClassifier(random_state=42).fit(xs_train, y_train)
print("Accuracy (scaled):", accuracy_score(y_test, sgd_scaled.predict(xs_test)))

# ---- Part 4: which digits get confused ----
off = conf_mx.copy().astype(float)
np.fill_diagonal(off, 0)
pairs = sorted(((int(off[i, j]), i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:8]
print("Top confusions (count, actual, predicted):", pairs)
row_sums = conf_mx.sum(axis=1, keepdims=True)
print("Per-digit recall:", {i: round(float(conf_mx[i, i] / row_sums[i, 0]), 3) for i in range(10)})

fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(conf_mx, display_labels=range(10)).plot(ax=ax, cmap="Blues", values_format="d", colorbar=True)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("SGD multiclass confusion matrix (MNIST test set)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
