"""
make_svm_from_gnn.py

Tworzy dataset pod SVM na bazie JUŻ POBRANYCH danych dla GNN:
- bierze tych samych użytkowników (nodes.jsonl)
- liczy followers/following z edges.csv
- bierze etykiety tagów z labels.npz + tag_vocab.json
- dociąga z HF API metadane modeli autora (downloads, likes, framework, trending_score)

Wymagania:
    pip install -U huggingface_hub numpy tqdm

Użycie:
    1) Ustaw GNN_DIR na katalog z plikami GNN (np. hf_gnn_data)
    2) python make_svm_from_gnn.py

Wyjście:
    svm_from_gnn/
      X.npy
      Y.npy
      usernames.json
      feature_names.json
      tag_vocab.json  (skopiowany dla spójności)
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm
from huggingface_hub import HfApi


# =========================
# KONFIGURACJA
# =========================
GNN_DIR = "hf_gnn_data"          # <- ustaw na katalog z outputem Twojego skryptu GNN
OUT_DIR = "svm_from_gnn"

PER_AUTHOR_MODELS_LIMIT = 50     # ile modeli per autor do agregacji statystyk
SLEEP_S = 0.0                    # ustaw np. 0.1 jeśli trafisz na rate limit
# =========================


def _sleep():
    if SLEEP_S and SLEEP_S > 0:
        import time
        time.sleep(SLEEP_S)


def load_gnn_nodes(gnn_dir: str) -> List[str]:
    path = os.path.join(gnn_dir, "nodes.jsonl")
    usernames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            usernames.append(obj["username"])
    return usernames


def load_gnn_edges_degrees(gnn_dir: str, num_nodes: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    edges.csv ma kolumny: src_id,dst_id,src_username,dst_username
    Interpretacja: src -> dst oznacza "src follows dst".
    - out_degree (following) = liczba wychodzących
    - in_degree (followers)  = liczba wchodzących
    """
    out_deg = np.zeros((num_nodes,), dtype=np.float32)
    in_deg = np.zeros((num_nodes,), dtype=np.float32)

    path = os.path.join(gnn_dir, "edges.csv")
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            s = int(row["src_id"])
            t = int(row["dst_id"])
            out_deg[s] += 1.0
            in_deg[t] += 1.0

    return in_deg, out_deg


def load_gnn_labels_and_vocab(gnn_dir: str) -> Tuple[np.ndarray, Dict[str, int]]:
    labels_path = os.path.join(gnn_dir, "labels.npz")
    vocab_path = os.path.join(gnn_dir, "tag_vocab.json")

    Y = np.load(labels_path)["Y"]  # uint8
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    # Upewniamy się, że Y jest 2D float32 dla klasyfikatorów
    Y = Y.astype(np.float32)
    return Y, vocab


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def infer_framework_from_tags(tags: List[str]) -> str:
    """
    Prosta heurystyka, gdy nie mamy library_name.
    """
    tagset = set(t.lower() for t in tags)
    # typowe tagi
    if "pytorch" in tagset or "torch" in tagset:
        return "pytorch"
    if "tensorflow" in tagset or "tf" in tagset:
        return "tensorflow"
    if "jax" in tagset or "flax" in tagset:
        return "jax"
    if "keras" in tagset:
        return "keras"
    if "onnx" in tagset:
        return "onnx"
    return "unknown"


def safe_getattr(obj, name: str, default=0.0):
    v = getattr(obj, name, None)
    if v is None:
        return default
    return v


def extract_author_model_stats(api: HfApi, author: str, limit: int) -> Dict[str, float | str]:
    """
    Agreguje statystyki po modelach autora:
    - n_models
    - sum/avg/max downloads
    - sum/avg/max likes
    - avg/max trending_score (jeśli dostępne)
    - framework (najczęstszy library_name albo heurystyka z tagów)
    """
    n = 0
    sum_dl = 0.0
    sum_likes = 0.0
    max_dl = 0.0
    max_likes = 0.0

    sum_tr = 0.0
    max_tr = 0.0
    tr_count = 0

    frameworks = []
    tags_all = []

    try:
        models_iter = api.list_models(author=author, sort="downloads", limit=limit)
        for m in models_iter:
            n += 1
            dl = float(safe_getattr(m, "downloads", 0.0) or 0.0)
            lk = float(safe_getattr(m, "likes", 0.0) or 0.0)
            sum_dl += dl
            sum_likes += lk
            max_dl = max(max_dl, dl)
            max_likes = max(max_likes, lk)

            # library_name bywa dostępne na nowszych obiektach ModelInfo
            lib = getattr(m, "library_name", None) or getattr(m, "libraryName", None)
            if isinstance(lib, str) and lib:
                frameworks.append(lib)

            mtags = getattr(m, "tags", None) or []
            for t in mtags:
                if isinstance(t, str) and t:
                    tags_all.append(t)

            tr = getattr(m, "trending_score", None)
            if tr is None:
                tr = getattr(m, "trendingScore", None)
            if tr is not None:
                try:
                    tr = float(tr)
                    sum_tr += tr
                    max_tr = max(max_tr, tr)
                    tr_count += 1
                except Exception:
                    pass

    except Exception:
        # autor może być chwilowo niedostępny / throttling
        pass

    avg_dl = sum_dl / n if n > 0 else 0.0
    avg_likes = sum_likes / n if n > 0 else 0.0
    avg_tr = (sum_tr / tr_count) if tr_count > 0 else 0.0

    if frameworks:
        framework = Counter(frameworks).most_common(1)[0][0]
    else:
        framework = infer_framework_from_tags(tags_all)

    return {
        "n_models": float(n),
        "sum_downloads": float(sum_dl),
        "avg_downloads": float(avg_dl),
        "max_downloads": float(max_dl),
        "sum_likes": float(sum_likes),
        "avg_likes": float(avg_likes),
        "max_likes": float(max_likes),
        "avg_trending_score": float(avg_tr),
        "max_trending_score": float(max_tr),
        "framework": framework,
    }


def one_hot_framework(frameworks: List[str]) -> Tuple[Dict[str, int], np.ndarray]:
    """
    Robi słownik frameworków i zwraca one-hot.
    """
    uniq = sorted(set(frameworks))
    vocab = {f: i for i, f in enumerate(uniq)}
    oh = np.zeros((len(frameworks), len(uniq)), dtype=np.float32)
    for i, fw in enumerate(frameworks):
        oh[i, vocab[fw]] = 1.0
    return vocab, oh


def main():
    ensure_dir(OUT_DIR)

    # --- wczytaj populację GNN
    usernames = load_gnn_nodes(GNN_DIR)
    N = len(usernames)

    in_deg, out_deg = load_gnn_edges_degrees(GNN_DIR, N)  # followers, following
    Y, tag_vocab = load_gnn_labels_and_vocab(GNN_DIR)

    # --- dociągnij statystyki autora (SVM features)
    token = os.getenv("HF_TOKEN", None)
    api = HfApi(token=token)

    stats_num = np.zeros((N, 9), dtype=np.float32)  # n_models + 6 stats + 2 trending stats
    frameworks = ["unknown"] * N

    for i, u in enumerate(tqdm(usernames, desc="Pobieranie cech SVM z HF API")):
        _sleep()
        s = extract_author_model_stats(api, u, PER_AUTHOR_MODELS_LIMIT)

        # followers/following jako cechy bazowe (wymagane w baseline) :contentReference[oaicite:4]{index=4}
        # Ale trzymamy je osobno, bo liczymy z grafu (edges.csv), a nie z API:
        # in_deg  = followers
        # out_deg = following

        stats_num[i, 0] = float(s["n_models"])
        stats_num[i, 1] = float(s["sum_downloads"])
        stats_num[i, 2] = float(s["avg_downloads"])
        stats_num[i, 3] = float(s["max_downloads"])
        stats_num[i, 4] = float(s["sum_likes"])
        stats_num[i, 5] = float(s["avg_likes"])
        stats_num[i, 6] = float(s["max_likes"])
        stats_num[i, 7] = float(s["avg_trending_score"])
        stats_num[i, 8] = float(s["max_trending_score"])
        frameworks[i] = str(s["framework"])

    fw_vocab, fw_oh = one_hot_framework(frameworks)

    # --- buduj X: [followers, following, (num stats...), framework one-hot]
    X = np.concatenate(
        [
            in_deg.reshape(-1, 1),     # followers
            out_deg.reshape(-1, 1),    # following
            stats_num,                 # modele/downloads/likes/trending
            fw_oh                      # framework
        ],
        axis=1
    ).astype(np.float32)

    feature_names = (
        ["followers_count", "following_count"]
        + [
            "n_models",
            "sum_downloads", "avg_downloads", "max_downloads",
            "sum_likes", "avg_likes", "max_likes",
            "avg_trending_score", "max_trending_score",
        ]
        + [f"framework::{fw}" for fw, _ in sorted(fw_vocab.items(), key=lambda kv: kv[1])]
    )

    # --- zapisz
    np.save(os.path.join(OUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUT_DIR, "Y.npy"), Y)  # to samo y co w GNN

    with open(os.path.join(OUT_DIR, "usernames.json"), "w", encoding="utf-8") as f:
        json.dump(usernames, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # kopiujemy vocab tagów dla spójności
    with open(os.path.join(OUT_DIR, "tag_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(tag_vocab, f, ensure_ascii=False, indent=2)

    with open(os.path.join(OUT_DIR, "framework_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(fw_vocab, f, ensure_ascii=False, indent=2)

    print("[OK] Zrobione.")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    print(f"  OUT_DIR: {OUT_DIR}")


if __name__ == "__main__":
    main()
