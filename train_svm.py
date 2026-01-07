"""
Wieloznaczny klasyfikator SVM z optymalizacją hiperparametrów Optuna.
Używa łańcuchów klasyfikatorów lub relevancji binarnej z obszernym dostrajaniem.
Wymagania:
    pip install numpy scikit-learn optuna matplotlib seaborn tqdm joblib
Użycie:
    python train_svm_optimized.py
"""

from __future__ import annotations
import json
import os
from typing import Dict, List, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, LinearSVC
from sklearn.multioutput import ClassifierChain, MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    hamming_loss,
    f1_score,
    precision_score,
    recall_score,
    jaccard_score,
    accuracy_score,
)
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
)
import joblib
import warnings

warnings.filterwarnings("ignore")

# =========================
# KONFIGURACJA
# =========================
DATA_DIR = "svm_from_gnn"
OUT_DIR = "svm_model"
N_TRIALS = 100  # Liczba prób Optuna
N_JOBS = -1  # Zadania równoległe (-1 = wszystkie rdzenie)
OPTUNA_TIMEOUT = 3600  # Maksymalny czas optymalizacji w sekundach (1 godzina)
CV_FOLDS = 3  # Foldy walidacji krzyżowej dla optymalizacji

TEST_SIZE = 0.2
VAL_SIZE = 0.1  # Podział walidacyjny z danych treningowych
RANDOM_STATE = 42
MIN_TAG_SAMPLES = 10  # Minimalna liczba próbek na tag

TOP_K_TAGS = 20  # Pokaż metryki dla top K tagów
SAVE_BEST_N_MODELS = 3  # Zapisz top N modeli z optymalizacji
# =========================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_data(
    data_dir: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    """Ładuj X, Y, nazwy użytkowników, nazwy cech i słownik tagów."""
    X = np.load(os.path.join(data_dir, "X.npy"))
    Y = np.load(os.path.join(data_dir, "Y.npy"))

    with open(os.path.join(data_dir, "usernames.json"), "r") as f:
        usernames = json.load(f)

    with open(os.path.join(data_dir, "feature_names.json"), "r") as f:
        feature_names = json.load(f)

    with open(os.path.join(data_dir, "tag_vocab.json"), "r") as f:
        tag_vocab = json.load(f)

    return X, Y, usernames, feature_names, tag_vocab


def analyze_label_distribution(Y: np.ndarray, tag_vocab: Dict[str, int]) -> Dict:
    """Analizuj rozkład etykiet i niezbalansowanie."""
    n_samples, n_tags = Y.shape
    tag_counts = Y.sum(axis=0)

    idx_to_tag = {v: k for k, v in tag_vocab.items()}

    stats = {
        "n_samples": n_samples,
        "n_tags": n_tags,
        "tag_counts": tag_counts,
        "avg_labels_per_sample": Y.sum(axis=1).mean(),
        "min_samples_per_tag": int(tag_counts.min()),
        "max_samples_per_tag": int(tag_counts.max()),
        "tags_with_few_samples": int(sum(tag_counts < MIN_TAG_SAMPLES)),
    }

    print("\n" + "=" * 60)
    print("ANALIZA ROZKŁADU ETYKIET")
    print("=" * 60)
    print(f"Całkowita liczba próbek: {stats['n_samples']}")
    print(f"Całkowita liczba tagów: {stats['n_tags']}")
    print(f"Średnia liczba etykiet na próbkę: {stats['avg_labels_per_sample']:.2f}")
    print(f"Minimalna liczba próbek na tag: {stats['min_samples_per_tag']}")
    print(f"Maksymalna liczba próbek na tag: {stats['max_samples_per_tag']}")
    print(f"Tagi z < {MIN_TAG_SAMPLES} próbkami: {stats['tags_with_few_samples']}")

    print(f"\nTop 10 najczęstszych tagów:")
    top_indices = np.argsort(tag_counts)[-10:][::-1]
    for idx in top_indices:
        tag_name = idx_to_tag[idx]
        count = int(tag_counts[idx])
        print(f" {tag_name}: {count} próbek ({100*count/n_samples:.1f}%)")

    return stats


def filter_rare_tags(
    Y: np.ndarray, tag_vocab: Dict[str, int], min_samples: int
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Usuń tagi z mniejszą niż min_samples liczbą wystąpień."""
    tag_counts = Y.sum(axis=0)
    valid_tags = tag_counts >= min_samples

    Y_filtered = Y[:, valid_tags]

    idx_to_tag = {v: k for k, v in tag_vocab.items()}
    new_vocab = {}
    new_idx = 0
    for old_idx, keep in enumerate(valid_tags):
        if keep:
            tag_name = idx_to_tag[old_idx]
            new_vocab[tag_name] = new_idx
            new_idx += 1

    removed = valid_tags.size - valid_tags.sum()
    print(f"\nUsunięto {removed} rzadkich tagów (< {min_samples} próbek)")
    print(f"Pozostałe tagi: {len(new_vocab)}")

    return Y_filtered, new_vocab


def create_classifier(trial: optuna.Trial, n_features: int, n_labels: int):
    """Utwórz klasyfikator z parametrami sugerowanymi przez Optuna."""

    strategy = trial.suggest_categorical(
        "strategy", ["binary_relevance", "classifier_chain", "ovr"]
    )
    classifier_type = trial.suggest_categorical(
        "classifier", ["linear_svc", "rbf_svc", "logistic"]
    )
    use_feature_selection = trial.suggest_categorical(
        "use_feature_selection", [True, False]
    )
    if use_feature_selection:
        n_features_to_select = trial.suggest_int(
            "n_features_to_select", max(5, n_features // 4), n_features
        )
    else:
        n_features_to_select = n_features

    scaler_type = trial.suggest_categorical("scaler", ["standard", "robust", "minmax"])

    if classifier_type == "linear_svc":
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        max_iter = trial.suggest_int("max_iter", 1000, 5000)

        base_clf = LinearSVC(
            C=C,
            class_weight=class_weight,
            max_iter=max_iter,
            dual="auto",
            random_state=RANDOM_STATE,
        )

    elif classifier_type == "rbf_svc":
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

        base_clf = SVC(
            C=C,
            kernel="rbf",
            gamma=gamma,
            class_weight=class_weight,
            probability=False,
            random_state=RANDOM_STATE,
            cache_size=500,
        )

    else:
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 2000)

        base_clf = LogisticRegression(
            C=C,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if strategy == "binary_relevance":
        clf = MultiOutputClassifier(base_clf, n_jobs=N_JOBS)
    elif strategy == "classifier_chain":
        order = trial.suggest_categorical("chain_order", ["random", "sorted"])
        if order == "random":
            clf = ClassifierChain(base_clf, order="random", random_state=RANDOM_STATE)
        else:
            clf = ClassifierChain(base_clf, order=None, random_state=RANDOM_STATE)
    else:
        clf = OneVsRestClassifier(base_clf, n_jobs=N_JOBS)

    return clf, scaler_type, n_features_to_select


def get_scaler(scaler_type: str):
    """Pobierz instancję skalera."""
    if scaler_type == "standard":
        return StandardScaler()
    elif scaler_type == "robust":
        return RobustScaler()
    else:
        return MinMaxScaler()


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> float:
    """Funkcja celu Optuna."""

    n_features = X_train.shape[1]
    n_labels = Y_train.shape[1]

    clf, scaler_type, n_features_to_select = create_classifier(
        trial, n_features, n_labels
    )

    scaler = get_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    if n_features_to_select < n_features:
        var_threshold = VarianceThreshold(threshold=0.0)
        X_train_scaled = var_threshold.fit_transform(X_train_scaled)
        X_val_scaled = var_threshold.transform(X_val_scaled)

        X_train_nonneg = X_train_scaled - X_train_scaled.min() + 1e-10

        feature_scores = np.zeros(X_train_nonneg.shape[1])
        for label_idx in range(Y_train.shape[1]):
            try:
                from sklearn.feature_selection import chi2 as chi2_score

                scores, _ = chi2_score(X_train_nonneg, Y_train[:, label_idx])
                feature_scores += scores
            except:
                pass

        if len(feature_scores) > n_features_to_select:
            top_k_indices = np.argsort(feature_scores)[-n_features_to_select:]
            X_train_scaled = X_train_scaled[:, top_k_indices]
            X_val_scaled = X_val_scaled[:, top_k_indices]

    try:
        clf.fit(X_train_scaled, Y_train)

        Y_pred = clf.predict(X_val_scaled)

        f1_micro = f1_score(Y_val, Y_pred, average="micro", zero_division=0)
        f1_macro = f1_score(Y_val, Y_pred, average="macro", zero_division=0)
        jaccard = jaccard_score(Y_val, Y_pred, average="samples", zero_division=0)

        score = 0.6 * f1_micro + 0.3 * f1_macro + 0.1 * jaccard

        trial.set_user_attr("f1_micro", f1_micro)
        trial.set_user_attr("f1_macro", f1_macro)
        trial.set_user_attr("jaccard", jaccard)

        return score

    except Exception as e:
        print(f"Próba nieudana: {e}")
        return 0.0


def optimize_hyperparameters(
    X_train: np.ndarray, Y_train: np.ndarray, X_val: np.ndarray, Y_val: np.ndarray
) -> optuna.Study:
    """Uruchom optymalizację Optuna."""
    print("\n" + "=" * 60)
    print("OPTYMALIZACJA HIPERPARAMETRÓW")
    print("=" * 60)
    print(f"Próby: {N_TRIALS}")
    print(f"Czas oczekiwania: {OPTUNA_TIMEOUT}s")
    print(f"Próbki treningowe: {X_train.shape[0]}")
    print(f"Próbki walidacyjne: {X_val.shape[0]}")
    print(f"Cechy: {X_train.shape[1]}")
    print(f"Etykiety: {Y_train.shape[1]}")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name="svm_multilabel_optimization",
    )

    study.optimize(
        lambda trial: objective(trial, X_train, Y_train, X_val, Y_val),
        n_trials=N_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=True,
        n_jobs=1,
    )

    print(f"\nOptymalizacja zakończona!")
    print(f"Najlepszy wynik: {study.best_value:.4f}")
    print(f"Najlepsze parametry:")
    for key, value in study.best_params.items():
        print(f" {key}: {value}")

    best_trial = study.best_trial
    print(f"\nMetryki najlepszej próby:")
    print(f" F1 (micro): {best_trial.user_attrs.get('f1_micro', 0):.4f}")
    print(f" F1 (macro): {best_trial.user_attrs.get('f1_macro', 0):.4f}")
    print(f" Jaccard: {best_trial.user_attrs.get('jaccard', 0):.4f}")

    return study


def train_final_model(
    X_train: np.ndarray, Y_train: np.ndarray, best_params: Dict[str, Any]
) -> Tuple[Any, Any, Any, Any]:
    """Trenuj ostateczny model z najlepszymi parametrami na pełnym zbiorze treningowym."""
    print("\n" + "=" * 60)
    print("TRENOWANIE OSTATECZNEGO MODELU")
    print("=" * 60)

    strategy = best_params["strategy"]
    classifier_type = best_params["classifier"]
    scaler_type = best_params["scaler"]

    if classifier_type == "linear_svc":
        base_clf = LinearSVC(
            C=best_params["C"],
            class_weight=best_params.get("class_weight"),
            max_iter=best_params["max_iter"],
            dual="auto",
            random_state=RANDOM_STATE,
        )
    elif classifier_type == "rbf_svc":
        base_clf = SVC(
            C=best_params["C"],
            kernel="rbf",
            gamma=best_params["gamma"],
            class_weight=best_params.get("class_weight"),
            random_state=RANDOM_STATE,
            cache_size=500,
        )
    else:
        base_clf = LogisticRegression(
            C=best_params["C"],
            class_weight=best_params.get("class_weight"),
            solver=best_params["solver"],
            max_iter=best_params["max_iter"],
            random_state=RANDOM_STATE,
            n_jobs=1,
        )

    if strategy == "binary_relevance":
        clf = MultiOutputClassifier(base_clf, n_jobs=N_JOBS)
    elif strategy == "classifier_chain":
        order = best_params.get("chain_order", "random")
        if order == "random":
            clf = ClassifierChain(base_clf, order="random", random_state=RANDOM_STATE)
        else:
            clf = ClassifierChain(base_clf, order=None, random_state=RANDOM_STATE)
    else:
        clf = OneVsRestClassifier(base_clf, n_jobs=N_JOBS)

    scaler = get_scaler(scaler_type)
    X_train_scaled = scaler.fit_transform(X_train)

    var_threshold = None
    top_k_indices = None

    if best_params.get("use_feature_selection", False):
        n_features_to_select = best_params["n_features_to_select"]

        var_threshold = VarianceThreshold(threshold=0.0)
        X_train_scaled = var_threshold.fit_transform(X_train_scaled)

        X_train_nonneg = X_train_scaled - X_train_scaled.min() + 1e-10
        feature_scores = np.zeros(X_train_nonneg.shape[1])

        for label_idx in range(Y_train.shape[1]):
            try:
                from sklearn.feature_selection import chi2 as chi2_score

                scores, _ = chi2_score(X_train_nonneg, Y_train[:, label_idx])
                feature_scores += scores
            except:
                pass

        if len(feature_scores) > n_features_to_select:
            top_k_indices = np.argsort(feature_scores)[-n_features_to_select:]
            X_train_scaled = X_train_scaled[:, top_k_indices]

    print("Trenowanie ostatecznego modelu...")
    clf.fit(X_train_scaled, Y_train)
    print("Trenowanie zakończone!")

    return clf, scaler, var_threshold, top_k_indices


def evaluate_model(
    clf: Any,
    scaler: Any,
    var_threshold: Any,
    top_k_indices: Any,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    tag_vocab: Dict[str, int],
) -> Dict:
    """Kompleksowa ewaluacja wytrenowanego modelu."""
    print("\n" + "=" * 60)
    print("EWALUACJA MODELU NA ZBIORZE TESTOWYM")
    print("=" * 60)

    X_test_scaled = scaler.transform(X_test)

    if var_threshold is not None:
        X_test_scaled = var_threshold.transform(X_test_scaled)

    if top_k_indices is not None:
        X_test_scaled = X_test_scaled[:, top_k_indices]

    Y_pred = clf.predict(X_test_scaled)

    hamming = hamming_loss(Y_test, Y_pred)
    f1_micro = f1_score(Y_test, Y_pred, average="micro", zero_division=0)
    f1_macro = f1_score(Y_test, Y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(Y_test, Y_pred, average="weighted", zero_division=0)
    f1_samples = f1_score(Y_test, Y_pred, average="samples", zero_division=0)
    precision_micro = precision_score(Y_test, Y_pred, average="micro", zero_division=0)
    recall_micro = recall_score(Y_test, Y_pred, average="micro", zero_division=0)
    jaccard = jaccard_score(Y_test, Y_pred, average="samples", zero_division=0)
    subset_acc = accuracy_score(Y_test, Y_pred)

    print(f"\nMetryki ogólne:")
    print(f" Strata Hamminga: {hamming:.4f}")
    print(f" Dokładność podzbioru: {subset_acc:.4f}")
    print(f" F1 Score (micro): {f1_micro:.4f}")
    print(f" F1 Score (macro): {f1_macro:.4f}")
    print(f" F1 Score (weighted): {f1_weighted:.4f}")
    print(f" F1 Score (samples): {f1_samples:.4f}")
    print(f" Precision (micro): {precision_micro:.4f}")
    print(f" Recall (micro): {recall_micro:.4f}")
    print(f" Wynik Jaccarda: {jaccard:.4f}")

    idx_to_tag = {v: k for k, v in tag_vocab.items()}
    tag_counts = Y_test.sum(axis=0)
    top_tag_indices = np.argsort(tag_counts)[-TOP_K_TAGS:][::-1]

    print(f"\nMetryki na tag (Top {TOP_K_TAGS} najczęstszych):")
    print(f"{'Tag':<30} {'Wsparcie':<10} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 76)

    per_tag_metrics = []
    for idx in top_tag_indices:
        tag_name = idx_to_tag[idx]
        support = int(tag_counts[idx])

        if support > 0:
            prec = precision_score(Y_test[:, idx], Y_pred[:, idx], zero_division=0)
            rec = recall_score(Y_test[:, idx], Y_pred[:, idx], zero_division=0)
            f1 = f1_score(Y_test[:, idx], Y_pred[:, idx], zero_division=0)

            print(
                f"{tag_name:<30} {support:<10} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}"
            )
            per_tag_metrics.append(
                {
                    "tag": tag_name,
                    "support": support,
                    "precision": float(prec),
                    "recall": float(rec),
                    "f1": float(f1),
                }
            )

    return {
        "hamming_loss": float(hamming),
        "subset_accuracy": float(subset_acc),
        "f1_micro": float(f1_micro),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "f1_samples": float(f1_samples),
        "precision_micro": float(precision_micro),
        "recall_micro": float(recall_micro),
        "jaccard_score": float(jaccard),
        "per_tag_metrics": per_tag_metrics,
    }


def save_results(
    clf: Any,
    scaler: Any,
    var_threshold: Any,
    top_k_indices: Any,
    tag_vocab: Dict[str, int],
    metrics: Dict,
    best_params: Dict,
    study: optuna.Study,
    out_dir: str,
):
    """Zapisz wszystkie wyniki."""
    ensure_dir(out_dir)

    model_data = {
        "classifier": clf,
        "scaler": scaler,
        "var_threshold": var_threshold,
        "top_k_indices": top_k_indices,
        "tag_vocab": tag_vocab,
        "best_params": best_params,
    }

    model_path = os.path.join(out_dir, "best_model.joblib")
    joblib.dump(model_data, model_path)

    metrics_path = os.path.join(out_dir, "test_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    study_path = os.path.join(out_dir, "optuna_study.joblib")
    joblib.dump(study, study_path)

    history = []
    for trial in study.trials:
        history.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            }
        )

    history_path = os.path.join(out_dir, "optimization_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Wyniki zapisane w {out_dir}/")
    print(f" - best_model.joblib")
    print(f" - test_metrics.json")
    print(f" - optuna_study.joblib")
    print(f" - optimization_history.json")


def plot_results(
    study: optuna.Study,
    metrics: Dict,
    tag_counts: np.ndarray,
    idx_to_tag: Dict[int, str],
    out_dir: str,
):
    """Generuj wizualizacje."""
    ensure_dir(out_dir)

    fig = plot_optimization_history(study)
    plot_path1 = os.path.join(out_dir, "optimization_history.png")
    fig.write_image(plot_path1)

    fig = plot_param_importances(study)
    plot_path2 = os.path.join(out_dir, "param_importances.png")
    fig.write_image(plot_path2)

    fig = plot_slice(study)
    plot_path3 = os.path.join(out_dir, "slice_plot.png")
    fig.write_image(plot_path3)

    plt.figure(figsize=(10, 6))
    sns.histplot(tag_counts, bins=50, kde=True)
    plt.title("Rozkład Liczby Próbek na Tag")
    plt.xlabel("Liczba Próbek")
    plt.ylabel("Liczba Tagów")
    plt.grid(True, alpha=0.3)
    plot_path4 = os.path.join(out_dir, "label_distribution.png")
    plt.savefig(plot_path4, dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 6))
    metric_names = [
        "F1 (micro)",
        "F1 (macro)",
        "F1 (weighted)",
        "Precision",
        "Recall",
        "Jaccard",
    ]
    metric_values = [
        metrics["f1_micro"],
        metrics["f1_macro"],
        metrics["f1_weighted"],
        metrics["precision_micro"],
        metrics["recall_micro"],
        metrics["jaccard_score"],
    ]
    bars = plt.bar(metric_names, metric_values, color="coral")
    plt.ylabel("Wynik")
    plt.title("Wydajność na Zbiorze Testowym")
    plt.ylim(0, 1)
    for bar, val in zip(bars, metric_values):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plot_path5 = os.path.join(out_dir, "test_performance.png")
    plt.savefig(plot_path5, dpi=150, bbox_inches="tight")
    plt.close()

    if metrics.get("per_tag_metrics"):
        top_tags = metrics["per_tag_metrics"][:10]
        tags = [m["tag"][:20] for m in top_tags]
        f1s = [m["f1"] for m in top_tags]
        plt.figure(figsize=(10, 6))
        plt.barh(tags, f1s, color="seagreen")
        plt.xlabel("Wynik F1")
        plt.title("Wyniki F1 dla Top 10 Tagów")
        plt.xlim(0, 1)
        plt.gca().invert_yaxis()
        plot_path6 = os.path.join(out_dir, "top_tags_f1.png")
        plt.savefig(plot_path6, dpi=150, bbox_inches="tight")
        plt.close()

    if metrics.get("per_tag_metrics"):
        df_metrics = pd.DataFrame(metrics["per_tag_metrics"])
        df_metrics.set_index("tag", inplace=True)
        df_metrics = df_metrics[["precision", "recall", "f1"]].head(20)
        plt.figure(figsize=(12, 8))
        sns.heatmap(df_metrics, annot=True, cmap="YlGnBu", fmt=".2f")
        plt.title("Heatmap Metryk na Tag (Top 20)")
        plot_path7 = os.path.join(out_dir, "per_tag_metrics_heatmap.png")
        plt.savefig(plot_path7, dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\n[OK] Wizualizacje zapisane w {out_dir}:")
    print(f" - optimization_history.png")
    print(f" - param_importances.png")
    print(f" - slice_plot.png")
    print(f" - label_distribution.png")
    print(f" - test_performance.png")
    print(f" - top_tags_f1.png")
    print(f" - per_tag_metrics_heatmap.png")


def main():
    print("=" * 60)
    print("PREDYKTOR WIELOZNACZNY SVM Z OPTYMALIZACJĄ OPTUNA")
    print("=" * 60)

    print(f"\nŁadowanie danych z {DATA_DIR}...")
    X, Y, usernames, feature_names, tag_vocab = load_data(DATA_DIR)

    stats = analyze_label_distribution(Y, tag_vocab)
    tag_counts = stats["tag_counts"]
    idx_to_tag = {v: k for k, v in tag_vocab.items()}

    Y_filtered, tag_vocab_filtered = filter_rare_tags(Y, tag_vocab, MIN_TAG_SAMPLES)

    print(f"\nPodział danych...")
    X_temp, X_test, Y_temp, Y_test = train_test_split(
        X, Y_filtered, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_temp, Y_temp, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE
    )

    print(f" Train: {X_train.shape[0]} próbek")
    print(f" Walidacja: {X_val.shape[0]} próbek")
    print(f" Test: {X_test.shape[0]} próbek")

    study = optimize_hyperparameters(X_train, Y_train, X_val, Y_val)

    X_train_full = np.vstack([X_train, X_val])
    Y_train_full = np.vstack([Y_train, Y_val])

    clf, scaler, var_threshold, top_k_indices = train_final_model(
        X_train_full, Y_train_full, study.best_params
    )

    metrics = evaluate_model(
        clf, scaler, var_threshold, top_k_indices, X_test, Y_test, tag_vocab_filtered
    )

    save_results(
        clf,
        scaler,
        var_threshold,
        top_k_indices,
        tag_vocab_filtered,
        metrics,
        study.best_params,
        study,
        OUT_DIR,
    )

    plot_results(study, metrics, tag_counts, idx_to_tag, OUT_DIR)

    print("\n" + "=" * 60)
    print("OPTYMALIZACJA I TRENOWANIE ZAKOŃCZONE!")
    print("=" * 60)
    print(f"\nOstateczny test F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"Ostateczny test F1 (macro): {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
