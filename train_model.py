"""
Predictive Maintenance - Training Pipeline
AI4I 2020 Dataset | Dual-head MLP | SMOTE + Class Weights | SHAP
"""
import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import shap

tf.random.set_seed(42)
np.random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────────────
DATA_PATH   = "data/ai4i2020.csv"
MODEL_DIR   = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 1. Load & Inspect ────────────────────────────────────────────────────────
print("="*60)
print("STEP 1: Loading dataset")
df = pd.read_csv(DATA_PATH)
print(f"  Shape: {df.shape}")
print(f"  Failure rate: {df['Machine failure'].mean()*100:.2f}%")

FAIL_COLS   = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAIL_LABELS = {0: "No Failure", 1: "TWF", 2: "HDF", 3: "PWF", 4: "OSF", 5: "RNF"}
FAIL_MAP    = {v: k for k, v in FAIL_LABELS.items()}

# ─── 2. Derive 6-class failure type label ────────────────────────────────────
print("\nSTEP 2: Deriving 6-class failure-type label")
def assign_failure_type(row):
    flags = row[FAIL_COLS]
    n = flags.sum()
    if n == 0:
        return 0  # No Failure
    if n == 1:
        return FAIL_MAP[flags.idxmax()]  # 1=TWF,2=HDF,3=PWF,4=OSF,5=RNF
    # Multiple flags – assign the dominant one (highest severity order)
    # Priority: TWF > HDF > PWF > OSF > RNF
    for col in FAIL_COLS:
        if row[col] == 1:
            return FAIL_MAP[col]
    return 0

df["failure_type"] = df.apply(assign_failure_type, axis=1)

print("  Failure type distribution:")
vc = df["failure_type"].value_counts().sort_index()
for idx, cnt in vc.items():
    print(f"    {idx} ({FAIL_LABELS[idx]}): {cnt}")

# ─── 3. Feature Engineering ──────────────────────────────────────────────────
print("\nSTEP 3: Feature engineering")
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
FEATURE_NAMES_DISPLAY = [
    "Air Temp (K)",
    "Process Temp (K)",
    "Rotational Speed (rpm)",
    "Torque (Nm)",
    "Tool Wear (min)"
]

# One-hot encode Type
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
type_encoded = ohe.fit_transform(df[["Type"]])
type_df = pd.DataFrame(type_encoded, columns=ohe.get_feature_names_out(["Type"]))

X_numeric = df[NUMERIC_FEATURES].values
X_type    = type_df.values
X_full    = np.hstack([X_type, X_numeric])   # Type_H, Type_L, Type_M + 5 numerics

# Targets
y_binary  = df["Machine failure"].values.astype(np.float32)
y_type    = df["failure_type"].values.astype(np.int32)

print(f"  Feature matrix shape: {X_full.shape}")

# ─── 4. Train / Test Split (stratified on 6-class label) ─────────────────────
print("\nSTEP 4: Stratified train/test split (80/20)")
X_train, X_test, yb_train, yb_test, yt_train, yt_test = train_test_split(
    X_full, y_binary, y_type,
    test_size=0.2, random_state=42, stratify=y_type
)
print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

# ─── 5. Scale numeric features ───────────────────────────────────────────────
print("\nSTEP 5: Scaling features")
# Scale only the numeric portion (columns 3 onward, after the 3 OHE type cols)
n_type_cols = type_encoded.shape[1]   # 3
scaler = StandardScaler()
X_train[:, n_type_cols:] = scaler.fit_transform(X_train[:, n_type_cols:])
X_test[:, n_type_cols:]  = scaler.transform(X_test[:, n_type_cols:])

# ─── 6. SMOTE on training set ────────────────────────────────────────────────
print("\nSTEP 6: SMOTE oversampling on training set")
print("  Before SMOTE:")
unique, counts = np.unique(yt_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"    {FAIL_LABELS[u]}: {c}")

# SMOTE on 6-class label
sm = SMOTE(random_state=42, k_neighbors=3)
X_res, yt_res = sm.fit_resample(X_train, yt_train)

# Reconstruct binary label from resampled type (0 = No Failure → 0, else 1)
yb_res = (yt_res != 0).astype(np.float32)

print("  After SMOTE:")
unique2, counts2 = np.unique(yt_res, return_counts=True)
for u, c in zip(unique2, counts2):
    print(f"    {FAIL_LABELS[u]}: {c}")
print(f"  Resampled shape: {X_res.shape}")

# One-hot encode the 6-class label for Head B
y_type_ohe_train = tf.keras.utils.to_categorical(yt_res, num_classes=6)
y_type_ohe_test  = tf.keras.utils.to_categorical(yt_test, num_classes=6)

# ─── 7. Build Dual-Head MLP ──────────────────────────────────────────────────
print("\nSTEP 7: Building dual-head MLP")
n_features = X_res.shape[1]

inp = layers.Input(shape=(n_features,), name="input")
x   = layers.Dense(64, activation="relu")(inp)
x   = layers.Dropout(0.2)(x)
x   = layers.Dense(32, activation="relu")(x)
x   = layers.Dropout(0.2)(x)

# Head A – binary failure detection
head_a = layers.Dense(1, activation="sigmoid", name="failure")(x)
# Head B – 6-class failure type
head_b = layers.Dense(6, activation="softmax", name="failure_type")(x)

model = Model(inputs=inp, outputs=[head_a, head_b])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={"failure": "binary_crossentropy", "failure_type": "categorical_crossentropy"},
    loss_weights={"failure": 1.0, "failure_type": 1.0},
    metrics={"failure": ["accuracy", keras.metrics.AUC(name="auc")],
             "failure_type": ["accuracy"]}
)
model.summary()

# ─── 8. Train ────────────────────────────────────────────────────────────────
print("\nSTEP 8: Fast Training")
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True,
                                   monitor="val_failure_auc", mode="max")
]

history = model.fit(
    X_res,
    {"failure": yb_res, "failure_type": y_type_ohe_train},
    validation_split=0.1,
    epochs=25,
    batch_size=128,
    callbacks=callbacks,
    verbose=0
)

# ─── 9. Evaluate ─────────────────────────────────────────────────────────────
print("\nSTEP 9: Evaluation on held-out test set")
pred_b_prob, pred_t_prob = model.predict(X_test, verbose=0)
pred_b = (pred_b_prob.ravel() > 0.5).astype(int)
pred_t = np.argmax(pred_t_prob, axis=1)

print("\n--- Binary Failure Detection ---")
print(classification_report(yb_test, pred_b,
      target_names=["No Failure", "Failure"], digits=4))

roc_auc = roc_auc_score(yb_test, pred_b_prob.ravel())
print(f"ROC-AUC: {roc_auc:.4f}")

# ─── 10. Fast SHAP Explainer ──────────────────────────────────────────────────
print("\nSTEP 10: Building fast SHAP explainer")

def model_predict_failure(X_in):
    prob, _ = model.predict(X_in, verbose=0)
    return prob.ravel()

# Background from training set (15 samples for extreme speed)
bg_idx = np.random.choice(len(X_res), size=15, replace=False)
background = X_res[bg_idx]

explainer = shap.KernelExplainer(model_predict_failure, background)


# Column names for all features
all_feature_names = list(ohe.get_feature_names_out(["Type"])) + NUMERIC_FEATURES

# ─── 11. Save everything ─────────────────────────────────────────────────────
print("\nSTEP 11: Saving artifacts")

# Keras model
model.save(os.path.join(MODEL_DIR, "model.keras"))
print("  Saved model.keras")

# Scaler & encoder
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "ohe.pkl"), "wb") as f:
    pickle.dump(ohe, f)
print("  Saved scaler.pkl, ohe.pkl")

# SHAP explainer
with open(os.path.join(MODEL_DIR, "shap_explainer.pkl"), "wb") as f:
    pickle.dump(explainer, f)
print("  Saved shap_explainer.pkl")

# Meta-info
meta = {
    "feature_names":         all_feature_names,
    "numeric_feature_names": NUMERIC_FEATURES,
    "display_names":         FEATURE_NAMES_DISPLAY,
    "fail_labels":           FAIL_LABELS,
    "n_type_cols":           int(n_type_cols),
    "type_categories":       ohe.categories_[0].tolist(),
    "thresholds": {
        "air_temp_K":   {"min": float(df["Air temperature [K]"].min()),
                         "max": float(df["Air temperature [K]"].max()),
                         "mean": float(df["Air temperature [K]"].mean())},
        "proc_temp_K":  {"min": float(df["Process temperature [K]"].min()),
                         "max": float(df["Process temperature [K]"].max()),
                         "mean": float(df["Process temperature [K]"].mean())},
        "rot_speed":    {"min": float(df["Rotational speed [rpm]"].min()),
                         "max": float(df["Rotational speed [rpm]"].max()),
                         "mean": float(df["Rotational speed [rpm]"].mean())},
        "torque":       {"min": float(df["Torque [Nm]"].min()),
                         "max": float(df["Torque [Nm]"].max()),
                         "mean": float(df["Torque [Nm]"].mean())},
        "tool_wear":    {"min": float(df["Tool wear [min]"].min()),
                         "max": float(df["Tool wear [min]"].max()),
                         "mean": float(df["Tool wear [min]"].mean())}
    },
    "roc_auc": round(roc_auc, 4)
}
with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print("  Saved meta.json")

# ─── 12. Evaluation plots ────────────────────────────────────────────────────
print("\nSTEP 12: Saving evaluation plots")

# Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
cm_b = confusion_matrix(yb_test, pred_b)
axes[0].imshow(cm_b, cmap="Blues")
axes[0].set_title("Binary Failure - Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm_b[i,j], ha="center", va="center", fontsize=14)
axes[0].set_xticks([0,1]); axes[0].set_yticks([0,1])
axes[0].set_xticklabels(["Pred: No", "Pred: Yes"])
axes[0].set_yticklabels(["True: No", "True: Yes"])

cm_t = confusion_matrix(yt_test, pred_t)
im = axes[1].imshow(cm_t, cmap="Blues")
axes[1].set_title("Failure Type - Confusion Matrix")
lbs = list(FAIL_LABELS.values())
axes[1].set_xticks(range(6)); axes[1].set_yticks(range(6))
axes[1].set_xticklabels(lbs, rotation=45)
axes[1].set_yticklabels(lbs)
for i in range(6):
    for j in range(6):
        axes[1].text(j, i, cm_t[i,j], ha="center", va="center", fontsize=9,
                     color="white" if cm_t[i,j] > cm_t.max()*0.5 else "black")
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrices.png"), dpi=120)
plt.close()

# Training history
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"], label="train")
axes[0].plot(history.history["val_loss"], label="val")
axes[0].set_title("Total Loss"); axes[0].legend()
axes[1].plot(history.history.get("failure_auc", []), label="train AUC")
axes[1].plot(history.history.get("val_failure_auc", []), label="val AUC")
axes[1].set_title("Binary Failure AUC"); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"), dpi=120)
plt.close()

print("\n" + "="*60)
print("TRAINING COMPLETE")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Artifacts saved to: {MODEL_DIR}/")
print("="*60)
