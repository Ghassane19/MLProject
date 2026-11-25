"""Comparaison étendue du livrable 2 sur des images réelles.

Exécution : python -m backend.livrable2
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectFromModel, SelectPercentile, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, matthews_corrcoef
from sklearn.model_selection import ParameterGrid, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from threadpoolctl import threadpool_limits

from backend.ml_pipeline import (
    CLASSES, FEATURE_CONFIG, LIMITATIONS, _half_features, _matrix, _save_data_figures,
    evaluate_predictions, prepare_manifest, split_development, write_json,
)

PROTOCOL_VERSION = "expanded-v2"
SELECTIONS = ("Toutes", "ANOVA_50pct", "Importance_RF_50pct")
MODELS = ("SVM", "SGD", "LogisticRegression", "RandomForest", "ExtraTrees",
          "kNN", "DecisionTree", "GaussianNB", "HistGradientBoosting", "MLP")
BALANCING = ("Sans_correction", "Correction_train")
RESAMPLED_MODELS = ("kNN", "GaussianNB", "HistGradientBoosting", "MLP")
MODEL_CATALOG = [
    {"model": "SVM", "principle": "Frontières non linéaires, noyau RBF"},
    {"model": "SGD", "principle": "Séparateur linéaire à marge, descente de gradient stochastique"},
    {"model": "LogisticRegression", "principle": "Régression logistique multiclasse"},
    {"model": "RandomForest", "principle": "Vote d'arbres ajustés sur échantillons bootstrap"},
    {"model": "ExtraTrees", "principle": "Ensemble d'arbres avec seuils aléatoires"},
    {"model": "kNN", "principle": "Vote des cinq voisins pondéré par distance"},
    {"model": "DecisionTree", "principle": "Règles de décision hiérarchiques"},
    {"model": "GaussianNB", "principle": "Modèle bayésien naïf gaussien"},
    {"model": "HistGradientBoosting", "principle": "Arbres construits par boosting sur histogrammes"},
    {"model": "MLP", "principle": "Réseau de neurones dense, une couche cachée"},
]
for entry in MODEL_CATALOG:
    entry["balancing"] = (
        "Suréchantillonnage du train en mémoire" if entry["model"] in RESAMPLED_MODELS
        else "Poids de classes calculés sur le train"
    )


def make_pipeline(selection, model, balancing, seed=42, parameters=None,
                  descriptor="HOG_HSV", memory=None):
    """Descripteurs fixes, puis transformations et sélection ajustées sur le train."""
    if selection not in SELECTIONS or model not in MODELS or balancing not in BALANCING:
        raise ValueError("Configuration d'expérience inconnue.")
    blocks = {"HOG_HSV": list(range(1860)), "HOG": list(range(1764)),
              "HSV": list(range(1764, 1860))}
    if descriptor not in blocks:
        raise ValueError("Descripteur inconnu.")
    selectors = {
        "Toutes": "passthrough",
        "ANOVA_50pct": SelectPercentile(f_classif, percentile=50),
        "Importance_RF_50pct": SelectFromModel(
            RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                   random_state=seed, n_jobs=4),
            threshold=-np.inf, max_features=_half_features,
        ),
    }
    weight = "balanced" if balancing == "Correction_train" else None
    estimators = {
        "SVM": SVC(C=10, kernel="rbf", class_weight=weight, random_state=seed, cache_size=512),
        "SGD": SGDClassifier(loss="hinge", alpha=0.0001, class_weight=weight,
                              random_state=seed, max_iter=2000, tol=1e-3),
        "LogisticRegression": LogisticRegression(C=1, class_weight=weight, max_iter=1500),
        "RandomForest": RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                               class_weight=weight, random_state=seed, n_jobs=4),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2,
                                           class_weight=weight, random_state=seed, n_jobs=4),
        "kNN": KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=4),
        "DecisionTree": DecisionTreeClassifier(max_depth=15, min_samples_leaf=3,
                                               class_weight=weight, random_state=seed),
        "GaussianNB": GaussianNB(var_smoothing=1e-9),
        # Pas de validation interne aléatoire qui ignorerait les groupes source.
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, l2_regularization=1, class_weight=None,
            early_stopping=False, random_state=seed),
        "MLP": MLPClassifier(hidden_layer_sizes=(96,), alpha=0.001, batch_size=128,
                             max_iter=180, tol=1e-3, n_iter_no_change=15,
                             early_stopping=False, random_state=seed),
    }
    estimator = estimators[model]
    if parameters:
        estimator.set_params(**parameters)
    return Pipeline([
        ("descriptor", ColumnTransformer([("block", "passthrough", blocks[descriptor])],
                                          remainder="drop")),
        ("variance", VarianceThreshold()),
        ("scaler", StandardScaler()),
        ("selection", selectors[selection]),
        ("model", estimator),
    ], memory=memory)


def training_sample(x, y, model, balancing, seed=42):
    """Duplication du train selon la stratégie retenue pour chaque modèle."""
    if model not in RESAMPLED_MODELS or balancing != "Correction_train":
        return x, y
    rng = np.random.default_rng(seed)
    labels, counts = np.unique(y, return_counts=True)
    indices = [np.arange(len(y))]
    for label, count in zip(labels, counts):
        if count < counts.max():
            indices.append(rng.choice(np.flatnonzero(y == label), int(counts.max() - count), replace=True))
    indices = np.concatenate(indices)
    return x[indices], y[indices]


def selected_indices(pipeline):
    """Indices dans les 1 860 composantes HOG + HSV d'origine."""
    descriptor = pipeline.named_steps.get("descriptor")
    original = np.asarray(descriptor.transformers_[0][2]) if descriptor is not None else np.arange(1860)
    indices = original[pipeline.named_steps["variance"].get_support()]
    selector = pipeline.named_steps["selection"]
    return indices if isinstance(selector, str) else indices[selector.get_support()]


def _configuration(row, parameters=None):
    result = {name: str(row[name]) for name in ("selection", "model", "balancing")}
    if parameters is not None:
        result["parameters"] = parameters
    return result


def _fit(pipeline, x, y, config, seed):
    fit_x, fit_y = training_sample(x, y, config["model"], config["balancing"], seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        pipeline.fit(fit_x, fit_y)
    diagnostics = {
        "fit_seconds": round(time.perf_counter() - started, 3), "fit_samples": len(fit_y),
        "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in caught),
        "warnings": " | ".join(dict.fromkeys(str(w.message) for w in caught)),
    }
    n_iter = getattr(pipeline.named_steps["model"], "n_iter_", None)
    diagnostics["iterations"] = int(np.max(n_iter)) if n_iter is not None else None
    return diagnostics


def _validation(pipeline, x_train, y_train, x_val, y_val):
    train_scores = evaluate_predictions(y_train, pipeline.predict(x_train))
    started = time.perf_counter()
    predicted = pipeline.predict(x_val)
    seconds = time.perf_counter() - started
    scores = evaluate_predictions(y_val, predicted)
    return {
        "train_accuracy": train_scores["accuracy"], "train_f1_macro": train_scores["f1_macro"],
        "validation_accuracy": scores["accuracy"],
        "validation_precision_weighted": scores["precision_weighted"],
        "validation_recall_weighted": scores["recall_weighted"],
        "validation_f1_macro": scores["f1_macro"], "validation_f1_weighted": scores["f1_weighted"],
        "validation_balanced_accuracy": scores["per_class"]["macro avg"]["recall"],
        "validation_mcc": float(matthews_corrcoef(y_val, predicted)),
        "validation_predict_seconds": round(seconds, 5),
    }


def grouped_learning_subsets(records, fractions=(0.25, 0.5, 0.75, 1.0), seed=42):
    """Sous-ensembles emboîtés, à groupes indivisibles et contenant les cinq classes."""
    if any(not 0 < f <= 1 for f in fractions) or list(fractions) != sorted(fractions):
        raise ValueError("Les fractions doivent être croissantes et dans ]0, 1].")
    rng = np.random.default_rng(seed)
    groups = list(rng.permutation(records.evaluation_group.unique()))
    initial = []
    for label in CLASSES:
        candidates = set(records.loc[records["class"].eq(label), "evaluation_group"])
        chosen = next(group for group in groups if group in candidates)
        if chosen not in initial:
            initial.append(chosen)
    order = initial + [group for group in groups if group not in initial]
    selected, cursor = set(initial), len(initial)
    for fraction in fractions:
        mask = records.evaluation_group.isin(selected).to_numpy()
        while mask.sum() < np.ceil(len(records) * fraction) and cursor < len(order):
            selected.add(order[cursor])
            cursor += 1
            mask = records.evaluation_group.isin(selected).to_numpy()
        yield fraction, np.flatnonzero(mask)


def group_bootstrap(labels, predicted, groups, n_resamples=1000, seed=42):
    """Bootstrap des groupes entiers, classes globales fixes, y compris les tirages incomplets."""
    labels, predicted, groups = np.asarray(labels), np.asarray(predicted), np.asarray(groups)
    if not (len(labels) == len(predicted) == len(groups)) or not len(labels):
        raise ValueError("Les observations, prédictions et groupes doivent être alignés et non vides.")
    if n_resamples < 2:
        raise ValueError("Au moins deux rééchantillonnages sont nécessaires.")
    codes, names = pd.factorize(groups)
    positions = {label: i for i, label in enumerate(CLASSES)}
    cluster_matrices = np.zeros((len(names), len(CLASSES), len(CLASSES)), dtype=np.int64)
    for code, actual, prediction in zip(codes, labels, predicted):
        cluster_matrices[code, positions[actual], positions[prediction]] += 1
    rng = np.random.default_rng(seed)
    bootstrap_scores, missing = [], 0
    for _ in range(n_resamples):
        matrix = cluster_matrices[rng.integers(0, len(names), size=len(names))].sum(axis=0)
        support = matrix.sum(axis=1)
        missing += int(np.any(support == 0))
        denominator = support + matrix.sum(axis=0)
        class_f1 = np.divide(2 * np.diag(matrix), denominator,
                             out=np.zeros(len(CLASSES), dtype=float), where=denominator != 0)
        bootstrap_scores.append([float(np.trace(matrix) / matrix.sum()), float(class_f1.mean())])
    original = evaluate_predictions(labels, predicted)
    return [
        {"metric": metric, "estimate": original[metric],
         "lower_95": float(np.percentile(np.array(bootstrap_scores)[:, i], 2.5)),
         "upper_95": float(np.percentile(np.array(bootstrap_scores)[:, i], 97.5)),
         "bootstrap_groups": len(names), "n_resamples": n_resamples,
         "absent_class_fraction": missing / n_resamples}
        for i, metric in enumerate(("accuracy", "f1_macro"))
    ]


def _parameter_grid(model):
    return {
        "SVM": {"C": [1, 10, 30]},
        "SGD": {"alpha": [1e-5, 1e-4, 1e-3]},
        "LogisticRegression": {"C": [0.1, 1, 10]},
        "RandomForest": {"min_samples_leaf": [1, 2], "max_depth": [20, None]},
        "ExtraTrees": {"min_samples_leaf": [1, 2, 4]},
        "kNN": {"n_neighbors": [3, 5, 9]},
        "DecisionTree": {"max_depth": [8, 15, None]},
        "GaussianNB": {"var_smoothing": [1e-9, 1e-7, 1e-5]},
        "HistGradientBoosting": {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]},
        "MLP": {"alpha": [0.0001, 0.001, 0.01]},
    }[model]


def _cross_validate(x, y, records, shortlist, output, seed, memory):
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    folds = list(splitter.split(x, y, records.evaluation_group))
    fold_rows = []
    for fold, (train, val) in enumerate(folds, 1):
        assert not set(records.iloc[train].evaluation_group) & set(records.iloc[val].evaluation_group)
        for label in CLASSES:
            train_count, val_count = int(np.sum(y[train] == label)), int(np.sum(y[val] == label))
            if not train_count or not val_count:
                raise ValueError("Une classe manque dans un pli ; revoir la validation groupée.")
            fold_rows.append({"fold": fold, "class": label,
                              "train_count": train_count, "validation_count": val_count})
    pd.DataFrame(fold_rows).to_csv(output / "cv_folds.csv", index=False)
    rows = []
    for config in shortlist:
        for parameters in ParameterGrid(_parameter_grid(config["model"])):
            for fold, (train, val) in enumerate(folds, 1):
                pipeline = make_pipeline(**config, parameters=parameters, seed=seed, memory=memory)
                diagnostics = _fit(pipeline, x[train], y[train], config, seed)
                scores = evaluate_predictions(y[val], pipeline.predict(x[val]))
                rows.append({**config, "parameters": json.dumps(parameters, sort_keys=True),
                             "fold": fold, "validation_f1_macro": scores["f1_macro"],
                             "validation_accuracy": scores["accuracy"], **diagnostics,
                             "train_samples": len(train), "validation_samples": len(val),
                             "validation_min_class_count": min(int(np.sum(y[val] == c)) for c in CLASSES)})
                pd.DataFrame(rows).to_csv(output / "cross_validation_results.csv", index=False)
            score = np.mean([r["validation_f1_macro"] for r in rows[-3:]])
            print(f"  CV {config['model']} {parameters} : F1 macro moyen = {score:.4f}", flush=True)
    summary = pd.DataFrame(rows).groupby(
        ["model", "selection", "balancing", "parameters"], sort=False, as_index=False,
    ).agg(mean_f1_macro=("validation_f1_macro", "mean"), std_f1_macro=("validation_f1_macro", "std"),
          mean_accuracy=("validation_accuracy", "mean"), n_folds=("fold", "count"),
          mean_fit_seconds=("fit_seconds", "mean"),
          convergence_warnings=("convergence_warning", "sum"))
    summary = summary.sort_values("mean_f1_macro", ascending=False, kind="stable").reset_index(drop=True)
    summary.to_csv(output / "cv_summary.csv", index=False)
    return summary, len(rows)


def run_experiments(root, output_dir=None, seed=42):
    root = Path(root).resolve()
    output = Path(output_dir) if output_dir else root / "backend/models/livrable2"
    output.mkdir(parents=True, exist_ok=True)
    # Un ancien rapport complet ne doit pas masquer l'état d'un nouveau calcul interrompu.
    write_json(output / "run_status.json", {"status": "running", "protocol_version": PROTOCOL_VERSION})
    with tempfile.TemporaryDirectory(prefix="ml-livrable2-cache-") as cache, threadpool_limits(limits=4):
        result = _run(root, output, seed, joblib.Memory(cache, verbose=0))
    write_json(output / "run_status.json", {"status": "complete", "protocol_version": PROTOCOL_VERSION})
    return result


def _run(root, output, seed, memory):
    dataset, figures = root / "dataset", output / "figures"
    figures.mkdir(exist_ok=True)
    print("1/8 — Audit, séparation groupée et extraction HOG + HSV", flush=True)
    manifest, audit = prepare_manifest(dataset)
    manifest, split_summary = split_development(manifest, seed)
    manifest.drop(columns="phash").to_csv(output / "split_manifest.csv", index=False)
    write_json(output / "quality_audit.json", audit)
    write_json(output / "split_summary.json", split_summary)
    _save_data_figures(dataset, manifest, figures, seed)
    development = manifest.loc[manifest.split.ne("test")].reset_index(drop=True)
    x, y = _matrix(dataset, development), development["class"].to_numpy()
    train_mask = development.split.eq("train").to_numpy()
    x_train, x_val, y_train, y_val = x[train_mask], x[~train_mask], y[train_mask], y[~train_mask]
    train_records = development.loc[train_mask].reset_index(drop=True)
    weights = compute_class_weight("balanced", classes=np.array(CLASSES), y=y_train)
    pd.DataFrame({"class": CLASSES,
                  "train_count": [int(np.sum(y_train == c)) for c in CLASSES],
                  "class_weight": weights,
                  "oversampled_train_count": int(pd.Series(y_train).value_counts().max())
                  }).to_csv(output / "balancing_details.csv", index=False)

    baseline_rows, dummy_scores = [], None
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=seed).fit(x_train, y_train)
        scores = evaluate_predictions(y_val, dummy.predict(x_val))
        baseline_rows.append({"stage": "validation", "strategy": strategy,
                              "accuracy": scores["accuracy"], "f1_macro": scores["f1_macro"]})
        if strategy == "most_frequent":
            dummy_scores = scores
    print("2/8 — 10 modèles × 3 sélections × 2 stratégies : 60 configurations", flush=True)
    comparisons, variable_rows = [], []
    for selection in SELECTIONS:
        for model in MODELS:
            for balancing in BALANCING:
                config = {"selection": selection, "model": model, "balancing": balancing}
                pipeline = make_pipeline(**config, seed=seed, memory=memory)
                diagnostics = _fit(pipeline, x_train, y_train, config, seed)
                scores = _validation(pipeline, x_train, y_train, x_val, y_val)
                indices = selected_indices(pipeline)
                row = {**config, "n_features": len(indices), **scores, **diagnostics}
                comparisons.append(row)
                variable_rows.extend({**config, "feature_index": int(i),
                                      "descriptor": "HOG" if i < 1764 else "HSV"} for i in indices)
                pd.DataFrame(comparisons).to_csv(output / "validation_results.csv", index=False)
                print(f"  {selection} / {model} / {balancing} : F1 = {scores['validation_f1_macro']:.4f}"
                      + (" [convergence à vérifier]" if diagnostics["convergence_warning"] else ""), flush=True)
    pd.DataFrame(variable_rows).to_csv(output / "selected_features_validation.csv", index=False)
    table = pd.DataFrame(comparisons).sort_values("validation_f1_macro", ascending=False, kind="stable")
    family_best = table.drop_duplicates("model")
    shortlist = [_configuration(row) for _, row in family_best.head(3).iterrows()]
    family_winners = {row.model: _configuration(row, parameters={}) for _, row in family_best.iterrows()}

    print("3/8 — Réglage des trois familles présélectionnées en validation croisée groupée", flush=True)
    cv_summary, cv_count = _cross_validate(x, y, development, shortlist, output, seed, memory)
    for model, row in cv_summary.drop_duplicates("model").set_index("model", drop=False).iterrows():
        family_winners[model] = _configuration(row, json.loads(row.parameters))
    winner = _configuration(cv_summary.iloc[0], json.loads(cv_summary.iloc[0].parameters))
    decision = {
        "winner": winner, "family_winners": family_winners,
        "screening_winner": _configuration(table.iloc[0]),
        "top_cv_models": [row["model"] for row in shortlist],
        "criterion": "cv_mean_f1_macro",
        "tie_break": "Ordre fixe des configurations en cas d'égalité exacte.",
        "frozen_before_test_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Meilleur candidat CV parmi les trois familles présélectionnées sur validation.",
    }
    write_json(output / "model_selection.json", decision)

    print("4/8 — Ablation des descripteurs et courbe d'apprentissage par groupes", flush=True)
    ablations, learning, learning_counts = [], [], []
    for descriptor in ("HOG", "HSV", "HOG_HSV"):
        config = {**winner, "selection": "Toutes"}
        pipeline = make_pipeline(**config, descriptor=descriptor, seed=seed, memory=memory)
        diagnostics = _fit(pipeline, x_train, y_train, config, seed)
        scores = evaluate_predictions(y_val, pipeline.predict(x_val))
        ablations.append({"descriptor": descriptor, "validation_f1_macro": scores["f1_macro"],
                          "validation_accuracy": scores["accuracy"],
                          "n_features": len(selected_indices(pipeline)), **diagnostics})
    pd.DataFrame(ablations).to_csv(output / "ablation_results.csv", index=False)
    for fraction, indices in grouped_learning_subsets(train_records, seed=seed):
        learning_counts.extend({"fraction": fraction, "class": label,
                                "count": int(np.sum(y_train[indices] == label))} for label in CLASSES)
        pipeline = make_pipeline(**winner, seed=seed, memory=memory)
        diagnostics = _fit(pipeline, x_train[indices], y_train[indices], winner, seed)
        scores = _validation(pipeline, x_train[indices], y_train[indices], x_val, y_val)
        learning.append({"fraction": fraction, "train_samples": len(indices),
                         "min_class_count": min(int(np.sum(y_train[indices] == c)) for c in CLASSES),
                         **scores, **diagnostics})
    pd.DataFrame(learning).to_csv(output / "learning_curve.csv", index=False)
    pd.DataFrame(learning_counts).to_csv(output / "learning_curve_class_counts.csv", index=False)

    print("5/8 — Réentraînement des dix finalistes, choix global déjà figé", flush=True)
    test = manifest.loc[manifest.split.eq("test")].reset_index(drop=True)
    x_test, y_test = _matrix(dataset, test), test["class"].to_numpy()
    predictions = test[["image", "class", "source", "evaluation_group"]].copy()
    test_scores, final_indices, timings, bootstrap_rows = {}, {}, [], []
    for model in MODELS:
        config = family_winners[model]
        pipeline = make_pipeline(**config, seed=seed, memory=memory)
        diagnostics = _fit(pipeline, x, y, config, seed)
        started = time.perf_counter()
        predicted = pipeline.predict(x_test)
        predict_seconds = time.perf_counter() - started
        predictions[model] = predicted
        scores = evaluate_predictions(y_test, predicted)
        scores["configuration"] = config
        scores["fit_diagnostics"] = diagnostics
        scores["balanced_accuracy"] = scores["per_class"]["macro avg"]["recall"]
        scores["mcc"] = float(matthews_corrcoef(y_test, predicted))
        scores["by_source"] = {
            str(source): evaluate_predictions(y_test[indices], predicted[indices])
            for source, indices in test.groupby("source").groups.items()
        }
        test_scores[model] = scores
        final_indices[model] = selected_indices(pipeline).tolist()
        timings.append({"model": model, "n_features": len(final_indices[model]), **diagnostics,
                        "test_predict_seconds": predict_seconds,
                        "batch_ms_per_image": predict_seconds * 1000 / len(test)})
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(np.array(scores["confusion_matrix"]), display_labels=CLASSES).plot(
            ax=ax, cmap="Blues", colorbar=False, xticks_rotation=25)
        ax.set_title(f"{model} — finaliste")
        fig.tight_layout()
        fig.savefig(figures / f"confusion_{model}.png", dpi=115)
        plt.close(fig)
        if model == winner["model"]:
            pipeline.memory = None
            joblib.dump({"pipeline": pipeline, "feature_config": FEATURE_CONFIG,
                         "class_names": list(CLASSES), "configuration": config,
                         "protocol_version": PROTOCOL_VERSION,
                         "note": "Probabilités éventuelles non calibrées ; utiliser pipeline.predict."},
                        output / "selected_pipeline.joblib", compress=3)
        bootstrap_rows.extend({"model": model, **row} for row in group_bootstrap(
            y_test, predicted, test.evaluation_group.to_numpy(), seed=seed))
        print(f"  {model} : accuracy = {scores['accuracy']:.4f}, F1 macro = {scores['f1_macro']:.4f}", flush=True)
    for strategy in ("most_frequent", "stratified"):
        dummy = DummyClassifier(strategy=strategy, random_state=seed).fit(x, y)
        scores = evaluate_predictions(y_test, dummy.predict(x_test))
        baseline_rows.append({"stage": "test", "strategy": strategy,
                              "accuracy": scores["accuracy"], "f1_macro": scores["f1_macro"]})
    pd.DataFrame(baseline_rows).to_csv(output / "baseline_results.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(timings).to_csv(output / "final_model_timings.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "bootstrap_intervals.csv", index=False)
    write_json(output / "selected_features_final.json", final_indices)

    print("6/8 — Contrôle de l'export sur les prédictions enregistrées", flush=True)
    artifact = joblib.load(output / "selected_pipeline.joblib")
    np.testing.assert_array_equal(artifact["pipeline"].predict(x_test), predictions[winner["model"]].to_numpy())
    versions = {name: importlib.metadata.version(name) for name in (
        "numpy", "pandas", "scikit-learn", "scikit-image", "Pillow", "matplotlib", "joblib")}
    result = {
        "protocol_version": PROTOCOL_VERSION, "run_status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "seed": seed, "versions": versions,
        "feature_config": FEATURE_CONFIG, "n_features": int(x.shape[1]),
        "selection": decision, "dummy_validation": dummy_scores,
        "split_summary": split_summary, "test": test_scores, "limitations": LIMITATIONS,
        "benchmark_count": len(comparisons), "cv_count": cv_count, "model_catalog": MODEL_CATALOG,
        "test_history": "Le test a déjà été évalué dans les versions antérieures du projet. "
                        "Les nouvelles décisions utilisent uniquement le développement. "
                        "Ces résultats restent exploratoires et demandent un nouveau test externe.",
        "cv_note": "La validation croisée règle les paramètres de trois familles présélectionnées "
                   "sur une validation issue du même développement. Elle n'est pas imbriquée et "
                   "ne fournit pas une estimation indépendante de généralisation.",
        "bootstrap_note": "Intervalles percentile 95 % sur 1 000 tirages de groupes entiers, "
                          "avec cinq classes fixes ; les tirages sans certaines classes sont inclus. "
                          "Seulement trois groupes de piétons : interprétation exploratoire, "
                          "sans conclusion de significativité statistique.",
        "timing_note": "Les temps de benchmark bénéficient d'un cache partagé des prétraitements ; "
                       "ils décrivent cette exécution, pas un classement universel de vitesse. "
                       "L'inférence est mesurée en batch, hors lecture d'image et extraction HOG/HSV.",
        "design": "60 configurations, puis réglage de trois familles sur trois plis groupés ; "
                  "ablation HOG/HSV et apprentissage par groupes après sélection ; "
                  "dix finalistes sur le test existant, avec bootstrap par groupes.",
    }
    write_json(output / "evaluation_metrics.json", result)
    print("7/8 — Export de la synthèse et de la traçabilité", flush=True)
    chosen = test_scores[winner["model"]]
    report = [
        "# Livrable 2 — Comparaison étendue", "",
        f"**{len(MODELS)} modèles, {len(comparisons)} configurations de présélection, "
        f"{cv_count} ajustements en validation croisée.**", "",
        f"Modèle retenu parmi les trois familles présélectionnées : **{winner['model']}**.",
        f"Sélection : {winner['selection']} ; correction : {winner['balancing']} ; "
        f"paramètres : {json.dumps(winner['parameters'], ensure_ascii=False)}.",
        f"F1 macro CV moyen : **{cv_summary.iloc[0].mean_f1_macro:.4f}** "
        f"(écart-type entre plis : {cv_summary.iloc[0].std_f1_macro:.4f}).",
        f"Accuracy test : **{chosen['accuracy']:.2%}** ; F1 macro test : **{chosen['f1_macro']:.4f}**.",
        "", "## Dix finalistes", "",
        "| Modèle | Accuracy test | F1 macro test |", "|---|---:|---:|",
        *[f"| {name} | {test_scores[name]['accuracy']:.2%} | {test_scores[name]['f1_macro']:.4f} |" for name in MODELS],
        "", "## Analyses complémentaires", "",
        "- Références majoritaire et aléatoire stratifiée.",
        "- ANOVA et importance Random Forest ; variables retenues exportées.",
        "- Déséquilibre traité sur les ajustements train seulement.",
        "- Réglage ciblé en validation croisée groupée ; effectifs de chaque pli publiés.",
        "- Ablation HOG / HSV / combinaison, à paramètres fixés et sans nouvelle sélection du gagnant.",
        "- Courbe d'apprentissage sur des sous-ensembles emboîtés à groupes indivisibles.",
        "- Temps d'apprentissage, dimensions, temps d'inférence en batch et diagnostics de convergence.",
        "- Précision/rappel/F1 par classe, erreurs visuelles et résultats par source.",
        "- Intervalles bootstrap par groupes, exploratoires.", "",
        "## Limites à défendre devant le jury", "", result["test_history"], "", result["cv_note"],
        "", "Les piétons de développement n'ont que trois groupes (531, 256 et 13 images). "
        "Les plis ne sont donc pas équivalents en support. L'écart-type CV décrit cette variabilité.",
        "", result["bootstrap_note"], "", result["timing_note"], "",
        *[f"- {note}" for note in LIMITATIONS], "",
        "## Suite du projet", "",
        "- Acquérir de nouveaux groupes et un test externe tenu à l'écart.",
        "- Étendre le réglage en validation imbriquée lorsque les groupes indépendants le permettent.",
        "- Vérifier les annotations et analyser les écarts de domaine et les erreurs difficiles.",
        "- Comparer un futur transfert d'apprentissage dans une expérience séparée et équitable.",
    ]
    (output / "RESULTATS.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"8/8 — Terminé : {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_experiments(args.root, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
