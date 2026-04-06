import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    matthews_corrcoef
)

import time
import os
RESULTS_DIR = "evaluation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Load data.pickle and model.p ────────────────────────────────────────────
print("Loading data and model...")
data_dict = pickle.load(open('./data.pickle', 'rb'))
model_dict = pickle.load(open('./model.p', 'rb'))

data   = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
model  = model_dict['model']

# ── Same split as train_classifier.py so test set is identical ──────────────
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

y_predict = model.predict(x_test)

times = []
for sample in x_test:
    start = time.perf_counter()
    model.predict([sample])
    times.append(time.perf_counter() - start)

print(f"Avg Inference Time : {np.mean(times)*1000:.3f} ms")
print(f"Max Inference Time : {np.max(times)*1000:.3f} ms")

# ════════════════════════════════════════════════════════════════════════════
# TEST 1 — Accuracy + Classification Report (Precision, Recall, F1)
# ════════════════════════════════════════════════════════════════════════════
print("\n════════════════════════════════════")
print("TEST 1 — Accuracy & Classification Report")
print("════════════════════════════════════")

acc = accuracy_score(y_test, y_predict)
mcc = matthews_corrcoef(y_test, y_predict)
print(f"Accuracy : {acc * 100:.2f}%")
print(f"MCC Score: {mcc:.4f}  (range -1 to +1, higher is better)\n")
print(classification_report(y_test, y_predict))

# Save as CSV
report_dict = classification_report(y_test, y_predict, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()
df_report.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print("Saved → evaluation_results/classification_report.csv")

# ════════════════════════════════════════════════════════════════════════════
# TEST 2 — Confusion Matrix
# ════════════════════════════════════════════════════════════════════════════
print("\n════════════════════════════════════")
print("TEST 2 — Confusion Matrix")
print("════════════════════════════════════")

cm   = confusion_matrix(y_test, y_predict, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

fig, ax = plt.subplots(figsize=(14, 14))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title("Confusion Matrix — SignSpeak Classifier")
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.close()
print("Saved → evaluation_results/confusion_matrix.png")

# ════════════════════════════════════════════════════════════════════════════
# TEST 3 — K-Fold Cross Validation
# ════════════════════════════════════════════════════════════════════════════
print("\n════════════════════════════════════")
print("TEST 3 — 5-Fold Cross Validation")
print("════════════════════════════════════")
print("Running... (this may take 1-2 minutes)")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, data, labels, cv=cv, scoring='f1_weighted')

print(f"F1 per fold : {cv_scores}")
print(f"Mean F1     : {cv_scores.mean():.4f}")
print(f"Std Dev     : {cv_scores.std():.4f}  (lower = more stable model)")

# ════════════════════════════════════════════════════════════════════════════
# TEST 4 — Learning Curve
# ════════════════════════════════════════════════════════════════════════════
print("\n════════════════════════════════════")
print("TEST 4 — Learning Curve")
print("════════════════════════════════════")
print("Running... (this may take 2-3 minutes)")

train_sizes, train_scores, val_scores = learning_curve(
    model, data, labels,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Accuracy', color='blue')
plt.plot(train_sizes, val_scores.mean(axis=1),   label='Validation Accuracy', color='orange')
plt.fill_between(train_sizes,
    train_scores.mean(axis=1) - train_scores.std(axis=1),
    train_scores.mean(axis=1) + train_scores.std(axis=1), alpha=0.1, color='blue')
plt.fill_between(train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.1, color='orange')
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.title("Learning Curve — SignSpeak Classifier")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "learning_curve.png"), dpi=300)
plt.close()
print("Saved → evaluation_results/learning_curve.png")

# ════════════════════════════════════════════════════════════════════════════
# TEST 5 — Feature Importance
# ════════════════════════════════════════════════════════════════════════════
print("\n════════════════════════════════════")
print("TEST 5 — Feature Importance")
print("════════════════════════════════════")

importances = model.feature_importances_
indices     = np.argsort(importances)[::-1]

plt.figure(figsize=(14, 5))
plt.bar(range(len(importances)), importances[indices], color='steelblue')
plt.title("Feature Importances — Random Forest (SignSpeak)")
plt.xlabel("Landmark Feature Index")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "feature_importance.png"), dpi=300)
plt.close()
print("Saved → evaluation_results/feature_importance.png")

# ════════════════════════════════════════════════════════════════════════════
print("\n✅ All tests complete. Files saved in your project folder.")