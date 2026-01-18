from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


@dataclass
class CFG:
    gnn_dir: str = "hf_gnn_data"
    pyg_path: str = "hf_gnn_data/pyg_data.pt"

    out_dir: str = "gnn_model"

    # Jeśli chcesz identyczny split jak SVM:
    # split_json: Optional[str] = "svm_model/split.json"  # ustaw None jeśli chcesz losowy split
    split_json: Optional[str] = None
    # Jeśli SVM filtrował rzadkie klasy:
    label_mask_npy: Optional[str] = (
        "svm_model/label_mask.npy"  # ustaw None, jeśli nie było filtracji
    )

    # Model: "sage" albo "gat"
    model_type: str = "gat"

    hidden: int = 448  # 800
    num_layers: int = 2
    dropout: float = 0.28557185904342947  # 0.2

    lr: float = 0.004998578343014432  # 1e-3
    weight_decay: float = 0.00010434075644093138  # 1e-4
    epochs: int = 6000
    patience: int = 30  # early stopping na val micro-F1

    # próg do multi-label
    threshold: float = 0.4920436087517883  # 0.3

    # if split_json=None
    test_size: float = 0.2
    val_size: float = 0.1
    seed: int = 42

    # optuna
    use_optuna: bool = True
    optuna_trials: int = 100
    out_root: str = "gnn_runs/"


def run_id() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


RUN_ID = run_id()
TAG_VOCAB_JSON = "svm_from_gnn/tag_vocab.json"
OPTUNA_DIR = f"gnn_runs/optuna_{RUN_ID}/"

# if not use optuna:
# CHECKPOINT_LOAD_DIR = "gnn_runs/optuna_2026-01-18_00-05-44"
CHECKPOINT_LOAD_DIR = "gnn_runs/optuna_2026-01-18_12-51-32"
