"""
train_svm_one_vs_rest.py

Trening baseline'u SVM dla multi-label klasyfikacji tagów na danych
wygenerowanych z GNN (svm_from_gnn/).

Wejście (OUT_DIR z make_svm_from_gnn.py):
  - X.npy  (float32)  shape [N, D]
  - Y.npy  (float32)  shape [N, C]  (multi-hot)
  - tag_vocab.json, feature_names.json, usernames.json (opcjonalnie do analizy)

Wymagania:
  pip install -U numpy scikit-learn joblib

Uruchomienie:
  python train_svm_one_vs_rest.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import joblib
import numpy as np

from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier


# =========================
# KONFIGURACJA
# =========================
@dataclass
class TrainConfig:
    data_dir: str = "svm_from_gnn"   # katalog z X.npy/Y.npy
    out_dir: str = "svm_model"       # gdzie zapisać model i wyniki

    test_size: float = 0.2
    val_size: float = 0.1            # liczona z pozostałej części po odjęciu testu
    random_state: int = 42

    # Filtrowanie klas:
    # jeśli jakiś tag jest ekstremalnie rzadki, SVM może się wywalać albo uczyć bez sensu
    min_pos_train: int = 5

    # Model:
    # "linearsvc" = solidny baseline; "sgd" = szybciej dla dużych danych
    model_type: str = "linearsvc"    # "linearsvc" albo "sgd"

    # Parametry LinearSVC:
    C: float = 1.0

    # Parametry SGD (hinge ~ linear SVM):
    sgd_alpha: float = 1e-4
    sgd_max_iter: int = 2000


CFG = TrainConfig()
# =========================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    X = np.load(os.path.join(data_dir, "X.npy")).astype(np.float32)
    Y = np.load(os.path.join(data_dir, "Y.npy")).astype(np.float32)

    vocab_path = os.path.join(data_dir, "tag_vocab.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        tag_vocab = json.load(f)

    # sanity check
    if X.ndim != 2:
        raise ValueError(f"X ma zły kształt: {X.shape} (powinno być 2D)")
    if Y.ndim != 2:
        raise ValueError(f"Y ma zły kształt: {Y.shape} (powinno być 2D)")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"N się nie zgadza: X {X.shape[0]} vs Y {Y.shape[0]}")

    return X, Y, tag_vocab


def build_estimator(cfg: TrainConfig):
    """
    Buduje pipeline: StandardScaler -> OneVsRest(SVM)
    """
    if cfg.model_type == "linearsvc":
        base = LinearSVC(C=cfg.C, class_weight=None)  # baseline, możesz dać class_weight="balanced"
    elif cfg.model_type == "sgd":
        base = SGDClassifier(
            loss="hinge",
            alpha=cfg.sgd_alpha,
            max_iter=cfg.sgd_max_iter,
            tol=1e-3,
            random_state=cfg.random_state,
        )
    else:
        raise ValueError("model_type musi być 'linearsvc' albo 'sgd'")

    clf = OneVsRestClassifier(base, n_jobs=-1)

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", clf),
    ])
    return pipe


def filter_labels_by_train_support(
    Y_train: np.ndarray, Y_val: np.ndarray, Y_test: np.ndarray, min_pos_train: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Usuwa kolumny (tagi), które mają za mało pozytywnych przykładów w TRAIN.
    Zwraca (mask, Y_train_f, Y_val_f, Y_test_f).
    """
    pos = (Y_train > 0.5).sum(axis=0)  # liczba jedynek per klasa
    mask = pos >= min_pos_train
    if mask.sum() == 0:
        raise ValueError("Po filtracji nie została żadna klasa. Zmniejsz min_pos_train.")
    return mask, Y_train[:, mask], Y_val[:, mask], Y_test[:, mask]


def evaluate_multilabel(y_true: np.ndarray, y_pred: np.ndarray, name: str):
    """
    y_true/y_pred: [N, C] w 0/1
    """
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro_p = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_r = recall_score(y_true, y_pred, average="micro", zero_division=0)

    print(f"\n=== {name} ===")
    print(f"micro-F1 : {micro_f1:.4f}")
    print(f"macro-F1 : {macro_f1:.4f}")
    print(f"micro-P  : {micro_p:.4f}")
    print(f"micro-R  : {micro_r:.4f}")

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
    }


def main():
    ensure_dir(CFG.out_dir)

    print("[1/5] Wczytuję dane...")
    X, Y, tag_vocab = load_data(CFG.data_dir)
    N, D = X.shape
    C = Y.shape[1]
    print(f"  X: {X.shape}  |  Y: {Y.shape}")

    # binarizacja (w razie float)
    Y_bin = (Y > 0.5).astype(np.int8)

    print("[2/5] Split train/val/test...")
    idx = np.arange(N)

    idx_trainval, idx_test = train_test_split(
        idx, test_size=CFG.test_size, random_state=CFG.random_state, shuffle=True
    )
    # val_size jest liczone względem trainval
    val_rel = CFG.val_size / (1.0 - CFG.test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_rel, random_state=CFG.random_state, shuffle=True
    )

    X_train, Y_train = X[idx_train], Y_bin[idx_train]
    X_val, Y_val = X[idx_val], Y_bin[idx_val]
    X_test, Y_test = X[idx_test], Y_bin[idx_test]

    print(f"  train: {X_train.shape[0]} | val: {X_val.shape[0]} | test: {X_test.shape[0]}")

    print("[3/5] Filtruję rzadkie tagi (żeby SVM miało z czego się uczyć)...")
    mask, Y_train_f, Y_val_f, Y_test_f = filter_labels_by_train_support(
        Y_train, Y_val, Y_test, min_pos_train=CFG.min_pos_train
    )
    kept = int(mask.sum())
    print(f"  Zachowane tagi: {kept}/{C} (min_pos_train={CFG.min_pos_train})")

    # zapis maski do późniejszej interpretacji tagów
    np.save(os.path.join(CFG.out_dir, "label_mask.npy"), mask)

    print("[4/5] Trening OneVsRest + SVM...")
    model = build_estimator(CFG)
    model.fit(X_train, Y_train_f)

    # predykcje: LinearSVC/SGD zwracają 0/1 przez .predict
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pred_test = model.predict(X_test)

    results = {
        "config": CFG.__dict__,
        "train": evaluate_multilabel(Y_train_f, pred_train, "TRAIN"),
        "val": evaluate_multilabel(Y_val_f, pred_val, "VAL"),
        "test": evaluate_multilabel(Y_test_f, pred_test, "TEST"),
        "N": int(N),
        "D": int(D),
        "C_original": int(C),
        "C_kept": int(kept),
    }

    print("[5/5] Zapisuję model i wyniki...")
    joblib.dump(model, os.path.join(CFG.out_dir, "svm_ovr_model.joblib"))

    split = {
        "idx_train": idx_train.tolist(),
        "idx_val": idx_val.tolist(),
        "idx_test": idx_test.tolist(),
    }
    with open(os.path.join(CFG.out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    with open(os.path.join(CFG.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Żeby łatwo mapować tagi po filtracji:
    # mask odnosi się do indeksów w Y (czyli do tag_vocab).
    with open(os.path.join(CFG.out_dir, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tag_vocab, f, ensure_ascii=False, indent=2)

    print(f"[OK] Gotowe. Wyniki i model w: {CFG.out_dir}")
    print(f"  Model: {os.path.join(CFG.out_dir, 'svm_ovr_model.joblib')}")
    print(f"  Wyniki: {os.path.join(CFG.out_dir, 'results.json')}")


if __name__ == "__main__":
    main()
