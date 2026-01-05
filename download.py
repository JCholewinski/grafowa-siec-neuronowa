"""
Pobieranie danych do GNN (Hugging Face Hub):
- seed autorów z top modeli (np. po downloads/likes/trending_score)
- budowa grafu użytkowników na podstawie relacji following (i opcjonalnie followers)
- zbudowanie etykiet multi-label: top-K tagów z modeli publikowanych przez użytkownika
- zapis do plików: nodes.jsonl, edges.csv, labels.npz, tag_vocab.json, pyg_data.pt (opcjonalnie)

Wymagania:
    pip install -U huggingface_hub requests tqdm numpy torch

Uwierzytelnienie (opcjonalnie, zalecane przy większym crawl):
    export HF_TOKEN=hf_xxx
albo:
    huggingface-cli login
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import requests
from tqdm import tqdm
import torch

from huggingface_hub import HfApi


# -----------------------------
# Konfiguracja i pomocnicze
# -----------------------------

@dataclass
class CrawlConfig:
    seed_models_limit: int = 500         # ile top modeli pobrać, żeby zebrać autorów
    seed_models_sort: str = "downloads"  # downloads | likes | trending_score | last_modified | created_at
    per_author_models_limit: int = 50    # ile modeli per autor do zbierania tagów
    max_hops: int = 2                    # 1 = tylko sąsiedzi seedów, 2 = sąsiedzi sąsiadów itd.
    max_neighbors_per_user: int = 200    # limit following/followers per user (żeby graf nie eksplodował)
    include_followers_edges: bool = False # jeżeli True: doda także krawędzie od followersów
    sleep_s: float = 0.0                # opcjonalny sleep między requestami (rate limiting)
    request_timeout_s: int = 30


def _safe_sleep(cfg: CrawlConfig):
    if cfg.sleep_s and cfg.sleep_s > 0:
        time.sleep(cfg.sleep_s)


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# -----------------------------
# Pobieranie danych z HF
# -----------------------------

def get_seed_authors(api: HfApi, cfg: CrawlConfig) -> List[str]:
    """
    Zbiera listę autorów na podstawie top modeli.
    W list_models sort można ustawić m.in. "downloads", "likes", "trending_score". :contentReference[oaicite:2]{index=2}
    """
    authors: List[str] = []
    seen: Set[str] = set()

    # Expand ogranicza payload i przyspiesza (zależnie od wersji hub)
    expand = ["author"]

    models_iter = api.list_models(
        sort=cfg.seed_models_sort,
        limit=cfg.seed_models_limit,
        expand=expand,
    )

    for m in models_iter:
        author = getattr(m, "author", None)
        if not author:
            continue
        # pomijamy "None"/puste i duplikaty
        if author not in seen:
            seen.add(author)
            authors.append(author)

    return authors


def iter_user_following(api: HfApi, username: str, cfg: CrawlConfig) -> List[str]:
    """
    Zwraca listę kont, które user obserwuje (following).
    W huggingface_hub funkcje user są paginowane. :contentReference[oaicite:3]{index=3}
    """
    following: List[str] = []
    try:
        # nazwa metody w huggingface_hub: list_user_following
        for u in api.list_user_following(username=username):
            if len(following) >= cfg.max_neighbors_per_user:
                break
            name = getattr(u, "name", None) or getattr(u, "username", None) or str(u)
            if name:
                following.append(name)
    except Exception:
        # user może nie istnieć / być ograniczony / chwilowy błąd
        return following
    return following


def iter_user_followers(api: HfApi, username: str, cfg: CrawlConfig) -> List[str]:
    """
    Zwraca listę followersów użytkownika.
    """
    followers: List[str] = []
    try:
        # nazwa metody w huggingface_hub: list_user_followers
        for u in api.list_user_followers(username=username):
            if len(followers) >= cfg.max_neighbors_per_user:
                break
            name = getattr(u, "name", None) or getattr(u, "username", None) or str(u)
            if name:
                followers.append(name)
    except Exception:
        return followers
    return followers


def get_author_tags_from_models(api: HfApi, author: str, cfg: CrawlConfig) -> List[str]:
    """
    Zbiera tagi z modeli opublikowanych przez autora.
    Używamy list_models(author=..., expand=["tags"]) i limitujemy liczbę modeli per autor.
    """
    tags: List[str] = []

    try:
        models_iter = api.list_models(
            author=author,
            sort="downloads",
            limit=cfg.per_author_models_limit,
            expand=["tags"],
        )
        for m in models_iter:
            mtags = getattr(m, "tags", None) or []
            # mtags bywa listą str
            for t in mtags:
                if isinstance(t, str) and t:
                    tags.append(t)
    except Exception:
        return tags

    return tags


# -----------------------------
# Budowa grafu i etykiet
# -----------------------------

def build_graph(api: HfApi, seed_users: List[str], cfg: CrawlConfig) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    BFS po sieci following/followers do max_hops.
    Krawędź: u -> v oznacza "u follows v".
    """
    nodes: Set[str] = set(seed_users)
    edges: Set[Tuple[str, str]] = set()

    frontier: Set[str] = set(seed_users)

    for hop in range(cfg.max_hops):
        next_frontier: Set[str] = set()
        for u in tqdm(sorted(frontier), desc=f"Hop {hop+1}/{cfg.max_hops}"):
            _safe_sleep(cfg)

            following = iter_user_following(api, u, cfg)
            for v in following:
                nodes.add(v)
                edges.add((u, v))
                next_frontier.add(v)

            if cfg.include_followers_edges:
                _safe_sleep(cfg)
                followers = iter_user_followers(api, u, cfg)
                for f in followers:
                    nodes.add(f)
                    edges.add((f, u))  # follower -> u (bo follower obserwuje u)
                    next_frontier.add(f)

        frontier = next_frontier

    return sorted(nodes), sorted(edges)


def build_label_matrix(
    api: HfApi,
    nodes: List[str],
    cfg: CrawlConfig,
    top_k_tags: int = 200,
    min_tag_freq: int = 5,
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, List[str]]]:
    """
    Buduje:
    - vocab tagów (top_k_tags po częstotliwości, z filtrem min_tag_freq)
    - macierz etykiet Y: shape [num_nodes, num_tags] (multi-hot)
    - surowe tagi per user (do debug/analizy)
    """
    tag_counts: Dict[str, int] = {}
    raw_user_tags: Dict[str, List[str]] = {}

    for u in tqdm(nodes, desc="Zbieranie tagów per autor"):
        _safe_sleep(cfg)
        tags = get_author_tags_from_models(api, u, cfg)
        raw_user_tags[u] = tags
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # filtr po freq + top_k
    items = [(t, c) for (t, c) in tag_counts.items() if c >= min_tag_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:top_k_tags]

    vocab = {t: i for i, (t, _) in enumerate(items)}
    num_nodes = len(nodes)
    num_tags = len(vocab)

    Y = np.zeros((num_nodes, num_tags), dtype=np.uint8)

    for i, u in enumerate(nodes):
        for t in raw_user_tags.get(u, []):
            j = vocab.get(t)
            if j is not None:
                Y[i, j] = 1

    return Y, vocab, raw_user_tags


# -----------------------------
# Zapis / eksport
# -----------------------------

def save_outputs(
    out_dir: str,
    nodes: List[str],
    edges: List[Tuple[str, str]],
    Y: np.ndarray,
    tag_vocab: Dict[str, int],
    raw_user_tags: Dict[str, List[str]],
):
    _ensure_dir(out_dir)

    # nodes.jsonl (id + username)
    nodes_path = os.path.join(out_dir, "nodes.jsonl")
    with open(nodes_path, "w", encoding="utf-8") as f:
        for idx, u in enumerate(nodes):
            f.write(json.dumps({"node_id": idx, "username": u}, ensure_ascii=False) + "\n")

    # edges.csv (src_id, dst_id)
    node2id = {u: i for i, u in enumerate(nodes)}
    edges_path = os.path.join(out_dir, "edges.csv")
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["src_id", "dst_id", "src_username", "dst_username"])
        for u, v in edges:
            if u in node2id and v in node2id:
                w.writerow([node2id[u], node2id[v], u, v])

    # labels + vocab
    npz_path = os.path.join(out_dir, "labels.npz")
    np.savez_compressed(npz_path, Y=Y)

    vocab_path = os.path.join(out_dir, "tag_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(tag_vocab, f, ensure_ascii=False, indent=2)

    raw_tags_path = os.path.join(out_dir, "raw_user_tags.json")
    with open(raw_tags_path, "w", encoding="utf-8") as f:
        json.dump(raw_user_tags, f, ensure_ascii=False)

    print(f"[OK] Zapisano:\n- {nodes_path}\n- {edges_path}\n- {npz_path}\n- {vocab_path}\n- {raw_tags_path}")


def export_pyg_data(out_dir: str, nodes: List[str], edges: List[Tuple[str, str]], Y: np.ndarray):
    """
    Opcjonalnie: eksport do formatu wygodnego dla PyTorch Geometric:
    - edge_index: [2, E]
    - y: [N, C]
    - x: proste cechy startowe (np. stopnie) — możesz potem zastąpić embeddingami / one-hot
    """
    if torch is None:
        print("[INFO] torch nie jest zainstalowany -> pomijam eksport PyG.")
        return

    node2id = {u: i for i, u in enumerate(nodes)}
    src = []
    dst = []
    for u, v in edges:
        if u in node2id and v in node2id:
            src.append(node2id[u])
            dst.append(node2id[v])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    y = torch.from_numpy(Y.astype(np.float32))

    # najprostsze cechy: out_degree i in_degree
    N = len(nodes)
    out_deg = np.zeros((N, 1), dtype=np.float32)
    in_deg = np.zeros((N, 1), dtype=np.float32)
    for s, t in zip(src, dst):
        out_deg[s, 0] += 1.0
        in_deg[t, 0] += 1.0
    x = torch.from_numpy(np.concatenate([out_deg, in_deg], axis=1))

    data = {"edge_index": edge_index, "x": x, "y": y, "num_nodes": N}
    out_path = os.path.join(out_dir, "pyg_data.pt")
    torch.save(data, out_path)
    print(f"[OK] Zapisano PyG tensor bundle: {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    # =========================
    # KONFIGURACJA (EDYTUJ TU)
    # =========================
    OUT_DIR = "hf_gnn_data"

    SEED_MODELS_LIMIT = 500
    SEED_MODELS_SORT = "downloads"   # downloads | likes | trending_score
    PER_AUTHOR_MODELS_LIMIT = 50

    MAX_HOPS = 2              # 0 = tylko seedy, 1 = sąsiedzi
    MAX_NEIGHBORS_PER_USER = 200
    INCLUDE_FOLLOWERS_EDGES = False

    TOP_K_TAGS = 200
    MIN_TAG_FREQ = 5

    SLEEP_S = 0.0                    # zwiększ jeśli trafisz na rate limit
    # =========================

    cfg = CrawlConfig(
        seed_models_limit=SEED_MODELS_LIMIT,
        seed_models_sort=SEED_MODELS_SORT,
        per_author_models_limit=PER_AUTHOR_MODELS_LIMIT,
        max_hops=MAX_HOPS,
        max_neighbors_per_user=MAX_NEIGHBORS_PER_USER,
        include_followers_edges=INCLUDE_FOLLOWERS_EDGES,
        sleep_s=SLEEP_S,
    )

    token = os.getenv("HF_TOKEN", None)
    api = HfApi(token=token)

    print("[1/4] Zbieram seed autorów z top modeli...")
    seed_users = get_seed_authors(api, cfg)
    print(f"  seed autorów: {len(seed_users)}")

    print("[2/4] Buduję graf użytkowników...")
    nodes, edges = build_graph(api, seed_users, cfg)
    print(f"  nodes: {len(nodes)} | edges: {len(edges)}")

    print("[3/4] Buduję etykiety (tagi)...")
    Y, vocab, raw_user_tags = build_label_matrix(
        api,
        nodes,
        cfg,
        top_k_tags=TOP_K_TAGS,
        min_tag_freq=MIN_TAG_FREQ,
    )
    print(f"  label matrix: {Y.shape} | vocab tags: {len(vocab)}")

    print("[4/4] Zapis danych...")
    save_outputs(OUT_DIR, nodes, edges, Y, vocab, raw_user_tags)
    export_pyg_data(OUT_DIR, nodes, edges, Y)


if __name__ == "__main__":
    main()