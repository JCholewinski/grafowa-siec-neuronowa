"""
train_gnn.py

Trening GNN (GraphSAGE lub GAT) dla multi-label node classification na danych z pyg_data.pt.

Wejście:
  - hf_gnn_data/pyg_data.pt  (z Twojego skryptu GNN)
  - opcjonalnie: svm_model/split.json (żeby mieć identyczny split jak w SVM)
  - opcjonalnie: svm_model/label_mask.npy (żeby trenować na tych samych tagach co SVM po filtracji)

Wyjście:
  - gnn_model/best_model.pt
  - gnn_model/results.json

Wymagania:
  pip install -U torch torch_geometric numpy scikit-learn
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
from datetime import datetime


from sklearn.metrics import f1_score, precision_score, recall_score

# PyG
from torch_geometric.nn import SAGEConv, GATConv
from torch_geometric.utils import add_self_loops

from gnn_configuration import (
    CFG,
    TAG_VOCAB_JSON,
    RUN_ID,
    OPTUNA_DIR,
    CHECKPOINT_LOAD_DIR,
)


CFG = CFG()


# =========================


def get_tag_names(num_labels: int) -> list[str]:
    """
    Zwraca listę nazw tagów w kolejności kolumn y.
    Jeśli jest tag_vocab.json i label_mask.npy – mapujemy jak w SVM.
    Jak nie ma – generujemy nazwy typu tag_0, tag_1, ...
    """
    # domyślne nazwy
    tag_names = [f"tag_{i}" for i in range(num_labels)]

    if not os.path.exists(TAG_VOCAB_JSON):
        return tag_names

    try:
        with open(TAG_VOCAB_JSON, "r", encoding="utf-8") as f:
            tag_vocab = json.load(f)  # {tag_name: idx}
        idx_to_tag_full = {idx: name for name, idx in tag_vocab.items()}

        # jeśli używasz label_mask (jak w SVM), musimy zmapować indeksy
        if CFG.label_mask_npy is not None and os.path.exists(CFG.label_mask_npy):
            mask = np.load(CFG.label_mask_npy)
            active_indices = [i for i, m in enumerate(mask) if m]
            tag_names = [idx_to_tag_full[i] for i in active_indices]
        else:
            # bez maski zakładamy, że idą 0..N-1
            tag_names = [idx_to_tag_full[i] for i in range(num_labels)]
    except Exception as e:
        print(f"[WARN] Nie udało się wczytać tag_vocab.json: {e}")

    # na wszelki wypadek przytnij do num_labels
    if len(tag_names) != num_labels:
        tag_names = tag_names[:num_labels]
        if len(tag_names) < num_labels:
            tag_names += [f"tag_{i}" for i in range(len(tag_names), num_labels)]

    return tag_names


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pyg_bundle(path: str):
    bundle = torch.load(path, map_location="cpu")
    # bundle: {"edge_index": [2,E], "x":[N,F], "y":[N,C], "num_nodes":N}
    return bundle


def make_masks_from_split(
    num_nodes: int, split_path: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with open(split_path, "r", encoding="utf-8") as f:
        split = json.load(f)

    train_idx = torch.tensor(split["idx_train"], dtype=torch.long)
    val_idx = torch.tensor(split["idx_val"], dtype=torch.long)
    test_idx = torch.tensor(split["idx_test"], dtype=torch.long)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask


def make_random_masks(num_nodes: int, test_size: float, val_size: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(num_nodes)
    rng.shuffle(idx)

    n_test = int(round(num_nodes * test_size))
    n_val = int(round(num_nodes * val_size))

    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True
    test_mask[torch.tensor(test_idx)] = True
    return train_mask, val_mask, test_mask


class GNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: int,
        num_layers: int,
        dropout: float,
        model_type: str,
    ):
        super().__init__()
        self.model_type = model_type
        self.dropout = dropout

        if num_layers < 2:
            raise ValueError("num_layers musi być >= 2")

        self.convs = nn.ModuleList()

        if model_type == "sage":
            self.convs.append(SAGEConv(in_dim, hidden))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden, hidden))
            self.convs.append(SAGEConv(hidden, out_dim))
        elif model_type == "gat":
            # prosta wersja GAT (jednogłowicowa na końcu)
            heads = 4
            self.convs.append(
                GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
                )
            self.convs.append(
                GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout)
            )
        else:
            raise ValueError("model_type musi być 'sage' albo 'gat'")

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x  # logits [N, C]


@torch.no_grad()
def evaluate(model: nn.Module, x, edge_index, y_true, mask, threshold: float):
    model.eval()
    logits = model(x, edge_index)[mask]
    y = y_true[mask]

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).cpu().numpy().astype(np.int8)
    y_np = (y > 0.5).cpu().numpy().astype(np.int8)

    micro_f1 = f1_score(y_np, pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_np, pred, average="macro", zero_division=0)
    micro_p = precision_score(y_np, pred, average="micro", zero_division=0)
    micro_r = recall_score(y_np, pred, average="micro", zero_division=0)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
    }


# ==============================
def infer_hidden_from_state_dict(state: dict) -> int:
    for k, v in state.items():
        if "convs.0" in k and k.endswith("weight") and v.dim() == 2:
            return v.size(0)
    raise ValueError(
        f"Nie można wykryć hidden. Dostępne klucze: {list(state.keys())[:10]}"
    )


def run_gnn_from_checkpoint(model_path: str) -> None:
    ensure_dir(CFG.out_dir)
    set_seed(CFG.seed)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    device = torch.device("cpu")
    print(f"[INFO] device: {device}")

    print("[1/4] Wczytuję pyg_data.pt ...")
    bundle = load_pyg_bundle(CFG.pyg_path)
    x = bundle["x"].float()
    edge_index = bundle["edge_index"].long()
    y = bundle["y"].float()
    num_nodes = int(bundle.get("num_nodes", x.size(0)))

    if x.size(0) != num_nodes or y.size(0) != num_nodes:
        raise ValueError("Niezgodne num_nodes vs x/y")

    if CFG.label_mask_npy is not None and os.path.exists(CFG.label_mask_npy):
        mask = np.load(CFG.label_mask_npy)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        y = y[:, mask_t]
        print(f"[INFO] Zastosowano label_mask: teraz y ma shape {tuple(y.shape)}")

    in_dim = x.size(1)
    out_dim = y.size(1)
    print(f"  x: {tuple(x.shape)} | y: {tuple(y.shape)} | edges: {edge_index.size(1)}")

    print("[2/4] Maski train/val/test ...")
    if CFG.split_json is not None and os.path.exists(CFG.split_json):
        train_mask, val_mask, test_mask = make_masks_from_split(
            num_nodes, CFG.split_json
        )
        print(f"[INFO] Używam splitu z: {CFG.split_json}")
    else:
        train_mask, val_mask, test_mask = make_random_masks(
            num_nodes, CFG.test_size, CFG.val_size, CFG.seed
        )
        print("[INFO] Używam losowego splitu")

    print(
        f"  train: {int(train_mask.sum())} | val: {int(val_mask.sum())} | test: {int(test_mask.sum())}"
    )

    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print("[3/4] Buduję model i wczytuję wagi ...")
    state = torch.load(model_path, map_location=device)

    hidden = infer_hidden_from_state_dict(state)
    CFG.hidden = hidden

    print(f"[INFO] Wykryto hidden z checkpointu: {hidden}")

    model = GNN(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=hidden,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout,
        model_type=CFG.model_type,
    ).to(device)

    model.load_state_dict(state)
    model.eval()

    print("[4/4] Ewaluacja ...")
    train_metrics = evaluate(model, x, edge_index, y, train_mask, CFG.threshold)
    val_metrics = evaluate(model, x, edge_index, y, val_mask, CFG.threshold)
    test_metrics = evaluate(model, x, edge_index, y, test_mask, CFG.threshold)

    results = {
        "config": CFG.__dict__,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "num_nodes": int(num_nodes),
        "in_dim": int(in_dim),
        "out_dim": int(out_dim),
        "model_path": model_path,
    }

    print("\n[INFO] Liczę metryki per-tag na teście...")

    with torch.no_grad():
        logits_test = model(x, edge_index)[test_mask]
        probs_test = torch.sigmoid(logits_test).cpu().numpy()
        y_test = (y[test_mask] > 0.5).cpu().numpy().astype(np.int8)
        y_pred = (probs_test >= CFG.threshold).astype(np.int8)

    tag_counts = y_test.sum(axis=0)
    num_labels = y_test.shape[1]
    tag_names = get_tag_names(num_labels)

    TOP_K_TAGS = min(20, num_labels)
    top_indices = np.argsort(tag_counts)[-TOP_K_TAGS:][::-1]

    from sklearn.metrics import precision_score, recall_score, f1_score

    per_tag_metrics = []
    print(f"\nMetryki per-tag (Top {TOP_K_TAGS}):")
    print(f"{'Tag':<30} {'Support':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 80)

    for idx in top_indices:
        support = int(tag_counts[idx])
        if support == 0:
            continue

        tag_name = tag_names[idx] if idx < len(tag_names) else f"tag_{idx}"
        prec = precision_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        rec = recall_score(y_test[:, idx], y_pred[:, idx], zero_division=0)
        f1 = f1_score(y_test[:, idx], y_pred[:, idx], zero_division=0)

        print(f"{tag_name:<30} {support:<10} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

        per_tag_metrics.append(
            {
                "tag": tag_name,
                "support": support,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        )

    results["test"]["per_tag_metrics"] = per_tag_metrics

    if per_tag_metrics:
        df_metrics = pd.DataFrame(per_tag_metrics).set_index("tag")[
            ["precision", "recall", "f1"]
        ]
        f_sum = df_metrics["f1"].sum()
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_metrics, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title(
            f"GNN - metryki per-tag (Top 20), model: {CFG.model_type}, f_sum: {f_sum:.3f}"
        )
        heatmap_path = os.path.join(CFG.out_dir, "per_tag_metrics_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
        if CFG.use_optuna:
            plt.savefig(
                f"{CFG.out_root}/{RUN_ID}/per_tag_metrics_heatmap.png",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close()
        print(f"[OK] Zapisano heatmapę per-tag: {heatmap_path}, f_sum: {f_sum}")

    with open(os.path.join(CFG.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n=== TRAIN ===", train_metrics)
    print("=== VAL   ===", val_metrics)
    print("=== TEST  ===", test_metrics)
    print(f"\n[OK] Gotowe. Model wczytany z: {model_path}")


def calculate_gnn(
    cfg: CFG, trial: optuna.Trial | None = None
) -> tuple[float, dict, dict]:
    set_seed(cfg.seed)
    device = "cpu"

    bundle = load_pyg_bundle(cfg.pyg_path)

    x = bundle["x"].float()
    edge_index = bundle["edge_index"].long()
    y = bundle["y"].float()
    num_nodes = x.size(0)

    if cfg.label_mask_npy and os.path.exists(cfg.label_mask_npy):
        mask = torch.tensor(np.load(cfg.label_mask_npy), dtype=torch.bool)
        y = y[:, mask]

    if cfg.split_json and os.path.exists(cfg.split_json):
        train_mask, val_mask, test_mask = make_masks_from_split(
            num_nodes, cfg.split_json
        )
    else:
        train_mask, val_mask, test_mask = make_random_masks(
            num_nodes, cfg.test_size, cfg.val_size, cfg.seed
        )

    x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
    train_mask, val_mask, test_mask = (
        train_mask.to(device),
        val_mask.to(device),
        test_mask.to(device),
    )

    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    model = GNN(
        x.size(1), y.size(1), cfg.hidden, cfg.num_layers, cfg.dropout, cfg.model_type
    ).to(device)

    pos_weight = (y.size(0) - y.sum(dim=0)) / (y.sum(dim=0) + 1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, x, edge_index, y, val_mask, cfg.threshold)
        val_micro_f1 = val_metrics["micro_f1"]

        # if trial:
        #     trial.report(val_micro_f1, epoch)
        #     if trial.should_prune():
        #         raise optuna.TrialPruned()

        if val_micro_f1 > best_val:
            best_val = val_micro_f1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.patience:
            break

    model.load_state_dict(best_state)

    train_f1 = evaluate(model, x, edge_index, y, train_mask, cfg.threshold)
    val_f1 = evaluate(model, x, edge_index, y, val_mask, cfg.threshold)
    test_f1 = evaluate(model, x, edge_index, y, test_mask, cfg.threshold)

    metrics = {"train_f1": train_f1, "val_f1": val_f1, "test_f1": test_f1}

    return best_val, best_state, metrics


# =========================
# OPTUNA
# =========================


def optuna_optimize(cfg: CFG) -> None:
    run_dir = os.path.join(cfg.out_root, f"optuna_{RUN_ID}")
    ensure_dir(run_dir)

    def objective(trial: optuna.Trial) -> float:
        heads = 4

        head_dim = trial.suggest_int("head_dim", 8, 128, step=8)
        hidden = head_dim * heads

        cfg.hidden = hidden
        cfg.dropout = trial.suggest_float("dropout", 0.1, 0.6)
        cfg.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        cfg.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        cfg.threshold = trial.suggest_float("threshold", 0.1, 0.5)

        trial.set_user_attr("hidden", hidden)
        trial.set_user_attr("heads", heads)
        trial.set_user_attr("head_dim", head_dim)

        best_val, _, _ = calculate_gnn(cfg, trial)

        return best_val

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=cfg.optuna_trials)

    with open(os.path.join(run_dir, "optuna_study.json"), "w") as f:
        json.dump(study.best_trial.params, f, indent=2)

    print("\n[OPTUNA] BEST PARAMS:", study.best_trial.params)

    for k, v in study.best_trial.params.items():
        setattr(cfg, k, v)

    best_val, best_state, metrics = calculate_gnn(cfg)

    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(
            {"best_params": study.best_trial.params, "metrics": metrics}, f, indent=2
        )

    print("[OK] Optuna zakończona. Wyniki zapisane w:", run_dir)


# =========================
# MAIN
# =========================


def main(optuna_dir) -> None:
    if CFG.use_optuna:
        optuna_optimize(CFG)
    else:
        optuna_dir = CHECKPOINT_LOAD_DIR

    run_gnn_from_checkpoint(f"{optuna_dir}/best_model.pt")


if __name__ == "__main__":
    main(OPTUNA_DIR)
