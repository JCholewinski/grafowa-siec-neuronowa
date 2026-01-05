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

from sklearn.metrics import f1_score, precision_score, recall_score

# PyG
from torch_geometric.nn import SAGEConv, GATConv


# =========================
# KONFIGURACJA
# =========================
@dataclass
class CFG:
    gnn_dir: str = "hf_gnn_data"
    pyg_path: str = "hf_gnn_data/pyg_data.pt"

    out_dir: str = "gnn_model"

    # Jeśli chcesz identyczny split jak SVM:
    split_json: Optional[str] = "svm_model/split.json"  # ustaw None jeśli chcesz losowy split
    # Jeśli SVM filtrował rzadkie klasy:
    label_mask_npy: Optional[str] = "svm_model/label_mask.npy"  # ustaw None, jeśli nie było filtracji

    # Model: "sage" albo "gat"
    model_type: str = "sage"

    hidden: int = 128
    num_layers: int = 2
    dropout: float = 0.3

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    patience: int = 20  # early stopping na val micro-F1

    # próg do multi-label
    threshold: float = 0.5

    # split losowy jeśli split_json=None
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42


CFG = CFG()
# =========================


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


def make_masks_from_split(num_nodes: int, split_path: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    val_idx = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True
    test_mask[torch.tensor(test_idx)] = True
    return train_mask, val_mask, test_mask


class GNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, num_layers: int, dropout: float, model_type: str):
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
            self.convs.append(GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden, hidden // heads, heads=heads, dropout=dropout))
            self.convs.append(GATConv(hidden, out_dim, heads=1, concat=False, dropout=dropout))
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


def main():
    ensure_dir(CFG.out_dir)
    set_seed(CFG.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print("[1/5] Wczytuję pyg_data.pt ...")
    bundle = load_pyg_bundle(CFG.pyg_path)
    x = bundle["x"].float()
    edge_index = bundle["edge_index"].long()
    y = bundle["y"].float()
    num_nodes = int(bundle.get("num_nodes", x.size(0)))

    if x.size(0) != num_nodes or y.size(0) != num_nodes:
        raise ValueError("Niezgodne num_nodes vs x/y")

    # jeśli SVM filtrował klasy, trenuj na tych samych
    if CFG.label_mask_npy is not None and os.path.exists(CFG.label_mask_npy):
        mask = np.load(CFG.label_mask_npy)
        mask_t = torch.tensor(mask, dtype=torch.bool)
        y = y[:, mask_t]
        print(f"[INFO] Zastosowano label_mask: teraz y ma shape {tuple(y.shape)}")

    in_dim = x.size(1)
    out_dim = y.size(1)
    print(f"  x: {tuple(x.shape)} | y: {tuple(y.shape)} | edges: {edge_index.size(1)}")

    print("[2/5] Maski train/val/test ...")
    if CFG.split_json is not None and os.path.exists(CFG.split_json):
        train_mask, val_mask, test_mask = make_masks_from_split(num_nodes, CFG.split_json)
        print(f"[INFO] Używam splitu z: {CFG.split_json}")
    else:
        train_mask, val_mask, test_mask = make_random_masks(num_nodes, CFG.test_size, CFG.val_size, CFG.seed)
        print("[INFO] Używam losowego splitu")

    print(f"  train: {int(train_mask.sum())} | val: {int(val_mask.sum())} | test: {int(test_mask.sum())}")

    # przeniesienie na device
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print("[3/5] Buduję model ...")
    model = GNN(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden=CFG.hidden,
        num_layers=CFG.num_layers,
        dropout=CFG.dropout,
        model_type=CFG.model_type,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    # multi-label: BCE z logitsami
    criterion = nn.BCEWithLogitsLoss()

    best_val = -1.0
    best_state = None
    bad_epochs = 0

    print("[4/5] Trening ...")
    for epoch in range(1, CFG.epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)
        loss = criterion(logits[train_mask], y[train_mask])

        loss.backward()
        optimizer.step()

        # ewaluacja co epokę (możesz robić co 5 dla speed)
        val_metrics = evaluate(model, x, edge_index, y, val_mask, CFG.threshold)
        val_micro = val_metrics["micro_f1"]

        if val_micro > best_val + 1e-6:
            best_val = val_micro
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | loss={loss.item():.4f} | val_microF1={val_micro:.4f}")

        if bad_epochs >= CFG.patience:
            print(f"[INFO] Early stopping na epoce {epoch} (best val_microF1={best_val:.4f})")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    print("[5/5] Test i zapis ...")
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
        "best_val_micro_f1": float(best_val),
    }

    print("\n=== TRAIN ===", train_metrics)
    print("=== VAL   ===", val_metrics)
    print("=== TEST  ===", test_metrics)

    torch.save(model.state_dict(), os.path.join(CFG.out_dir, "best_model.pt"))
    with open(os.path.join(CFG.out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Zapisano: {CFG.out_dir}/best_model.pt oraz results.json")


if __name__ == "__main__":
    main()
