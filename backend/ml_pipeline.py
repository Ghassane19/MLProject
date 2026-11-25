"""Outils partagés : audit, séparation, HOG + HSV et inférence.

Exécution : python -m backend.ml_pipeline
Les images originales ne sont jamais modifiées. Les scènes de démonstration
ne participent ni à l'entraînement ni à l'évaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


CLASSES = ("bus", "car", "cyclist", "motorcycle", "pedestrian")
FEATURE_CONFIG = {
    "size": [128, 128], "padding_rgb": [114, 114, 114],
    "hog_orientations": 9, "hog_pixels_per_cell": [16, 16],
    "hog_cells_per_block": [2, 2], "hog_block_norm": "L2-Hys",
    "hsv_bins_per_channel": 32,
}
LIMITATIONS = [
    "Les étiquettes et les relations entre sources ne sont pas certifiées exhaustivement.",
    "Le pHash signale des ressemblances, sans prouver que deux objets sont identiques.",
    "Les groupes source sont indivisibles : 15 % par classe n'est pas toujours réalisable.",
    "Quelques grands groupes de piétons limitent la représentativité de la validation ; "
    "l'écart de domaine entre sources peut influencer les scores.",
    "Les images apparentées restent dans leur groupe ; les observations du test ne sont "
    "donc pas toutes indépendantes. Les scores ne prouvent pas une robustesse en circulation réelle.",
    "Les probabilités éventuellement fournies ne sont pas calibrées pour un usage réel.",
    "Les photos de produit, les fonds et les recadrages peuvent créer des raccourcis visuels.",
    "Les explications des confusions sont des hypothèses à vérifier sur les images.",
]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_pixels(image_path, config=None):
    config = FEATURE_CONFIG if config is None else config
    with Image.open(image_path) as image:
        image = ImageOps.pad(
            ImageOps.exif_transpose(image).convert("RGB"), tuple(config["size"]),
            method=Image.Resampling.LANCZOS, color=tuple(config["padding_rgb"]),
        )
        return np.asarray(image, dtype=np.float32) / 255.0


def extract_features(image_path, config=None):
    """Descripteur fixe, sans ajustement sur les données ni accès aux étiquettes."""
    config = FEATURE_CONFIG if config is None else config
    pixels = load_pixels(image_path, config)
    gradient = hog(
        rgb2gray(pixels), orientations=config["hog_orientations"],
        pixels_per_cell=tuple(config["hog_pixels_per_cell"]),
        cells_per_block=tuple(config["hog_cells_per_block"]),
        block_norm=config["hog_block_norm"], feature_vector=True,
    )
    hsv = rgb2hsv(pixels)
    histograms = []
    for channel in range(3):
        counts, _ = np.histogram(hsv[:, :, channel], bins=config["hsv_bins_per_channel"], range=(0, 1))
        histograms.append(counts.astype(np.float32) / counts.sum())
    result = np.concatenate([gradient, *histograms]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Caractéristiques non finies : {image_path}")
    return result


def _inspect_image(dataset, row):
    path = dataset / row["image"]
    try:
        content = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            gray = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
            coefficients = dctn(gray, type=2, norm="ortho")[:8, :8].ravel()
            phash = int("".join("1" if v else "0" for v in coefficients > np.median(coefficients[1:])), 2)
        return {**row, "md5": hashlib.md5(content).hexdigest(), "actual_sha256": digest, "phash": phash}
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return {**row, "error": str(error)}


def prepare_manifest(dataset):
    """Recontrôle la provenance, filtre en mémoire et relie les ressemblances par prudence."""
    dataset = Path(dataset).resolve()
    metadata = pd.read_csv(dataset / "sources.csv", keep_default_na=False)
    required = {"image", "class", "split", "source_group", "pixel_sha256", "source"}
    if not required.issubset(metadata.columns) or metadata.image.duplicated().any():
        raise ValueError("Métadonnées manquantes ou chemins répétés dans sources.csv.")
    records = metadata.loc[metadata.split.isin(["train", "validation", "test"])].copy()
    if records.empty or not set(records["class"]).issubset(CLASSES) or records.source_group.eq("").any():
        raise ValueError("Classes ou groupes source invalides.")
    actual = set()
    for split in ("train", "validation", "test"):
        folder = dataset / split
        if folder.exists():
            actual.update(p.relative_to(dataset).as_posix() for p in folder.rglob("*") if p.is_file())
    if actual != set(records.image):
        raise ValueError("Les fichiers et sources.csv ne correspondent plus. Refaire l'audit des données.")
    for row in records.to_dict("records"):
        path = (dataset / row["image"]).resolve()
        expected_parts = (row["split"], row["class"])
        if not path.is_relative_to(dataset) or path.relative_to(dataset).parts[:2] != expected_parts:
            raise ValueError(f"Chemin ou étiquette incohérent : {row['image']}")
    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(pool.map(lambda row: _inspect_image(dataset, row), records.to_dict("records")))
    excluded = [{"image": r["image"], "reason": "illisible", "detail": r["error"]} for r in inspected if "error" in r]
    readable = [r for r in inspected if "error" not in r]
    for row in readable:
        if row["actual_sha256"] != row["pixel_sha256"]:
            raise ValueError(f"Pixels différents de la provenance : {row['image']}")
    priority = {"test": 0, "validation": 1, "train": 2}
    retained, seen = [], {}
    for row in sorted(readable, key=lambda r: (priority[r["split"]], r["image"])):
        digest = row["actual_sha256"]
        if digest in seen:
            if seen[digest]["class"] != row["class"]:
                raise ValueError("Doublon exact avec étiquettes contradictoires ; revue nécessaire.")
            excluded.append({"image": row["image"], "reason": "doublon exact", "kept": seen[digest]["image"]})
        else:
            retained.append(row)
            seen[digest] = row
    if not retained:
        raise ValueError("Aucune image exploitable après le contrôle de lisibilité.")
    frame = pd.DataFrame(retained).sort_values("image").reset_index(drop=True)
    if frame.groupby("source_group").split.nunique().max() > 1:
        raise ValueError("Groupes source partagés entre splits, ou aucune image exploitable.")
    parent = {name: name for name in frame.source_group.unique()}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    pairs = []
    rows = frame.to_dict("records")
    for i, row in enumerate(rows):
        for other in rows[:i]:
            distance = (row["phash"] ^ other["phash"]).bit_count()
            if distance <= 5:
                if row["split"] != other["split"]:
                    raise ValueError("Ressemblance pHash entre splits : revue requise avant entraînement.")
                pairs.append({"image_a": other["image"], "image_b": row["image"], "distance": distance})
                left, right = find(row["source_group"]), find(other["source_group"])
                parent[max(left, right)] = min(left, right)
    frame["evaluation_group"] = frame.source_group.map(find)
    audit = {
        "input_images": len(records), "retained_images": len(frame), "excluded": excluded,
        "near_pairs": pairs, "phash_max_distance": 5,
        "quality_decisions": "Conservation des images sombres et des paires ressemblantes. "
        "Les groupes apparentés sont réunis par prudence pour la validation, sans "
        "affirmer qu'il s'agit de doublons ni modifier les originaux.",
        "label_review": "Revue visuelle partielle héritée du notebook de nettoyage.",
    }
    return frame, audit


def split_development(frame, seed=42, fraction=0.15):
    """Cherche des effectifs proches de la cible, avec des groupes indivisibles.

    La règle de support est fixée avant apprentissage : au moins la moitié de
    la cible par classe en validation et la moitié des exemples en train.
    Elle évite une validation de seulement 13 piétons quand un groupe de 256
    est disponible. Aucune performance n'intervient dans ce choix.
    """
    frame = frame.copy()
    train = frame[frame.split == "train"]
    if (frame.split == "validation").any():
        selected = frame
        method = "Dossier validation existant, contrôlé sans partage de groupes."
    else:
        labels = train["class"].to_numpy()
        totals = Counter(labels)
        best = None
        candidates = GroupShuffleSplit(n_splits=1024, test_size=fraction, random_state=seed)
        for train_indices, validation_indices in candidates.split(train, labels, train.evaluation_group):
            counts = Counter(labels[validation_indices])
            if any(counts[c] < max(1, int(np.ceil(totals[c] * fraction / 2)))
                   or counts[c] > totals[c] / 2 for c in CLASSES):
                continue
            error = sum(((counts[c] - totals[c] * fraction) / totals[c]) ** 2 for c in CLASSES)
            if best is None or error < best[0]:
                best = (error, validation_indices)
        if best is None:
            raise ValueError("Aucune validation groupée avec un support suffisant pour les cinq classes.")
        selected = frame.copy()
        selected.loc[train.iloc[best[1]].index, "split"] = "validation"
        method = "1024 partitions de groupes à graine fixe ; choix par effectifs uniquement. "
        method += "Support validation minimal = moitié de la cible 15 % par classe ; train >= 50 % par classe."
    if selected.groupby("evaluation_group").split.nunique().max() > 1:
        raise ValueError("Fuite de groupes entre entraînement, validation et test.")
    for split in ("train", "validation", "test"):
        if set(selected.loc[selected.split == split, "class"]) != set(CLASSES):
            raise ValueError(f"Les cinq classes doivent être présentes dans {split}.")
    counts = selected.groupby(["split", "class"]).size().rename("count").reset_index()
    summary = {
        "seed": seed, "target_validation_fraction_of_development": fraction, "method": method,
        "counts": counts.to_dict("records"), "group_overlap": 0,
        "groups_per_split": selected.groupby("split").evaluation_group.nunique().to_dict(),
        "actual_validation_fraction": float((selected.split == "validation").sum() / (selected.split != "test").sum()),
    }
    return selected.reset_index(drop=True), summary


def _matrix(dataset, records):
    with ThreadPoolExecutor(max_workers=4) as pool:
        vectors = list(pool.map(lambda name: extract_features(dataset / name), records.image))
    return np.stack(vectors)


def _half_features(x):
    """Fonction sérialisable pour la sélection de la moitié des variables."""
    return max(1, x.shape[1] // 2)


def _pipeline(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def evaluate_predictions(labels, predicted):
    report = classification_report(labels, predicted, labels=list(CLASSES), output_dict=True, zero_division=0)
    matrix = confusion_matrix(labels, predicted, labels=list(CLASSES))
    pairs = []
    for i, left in enumerate(CLASSES):
        for j in range(i + 1, len(CLASSES)):
            count = int(matrix[i, j] + matrix[j, i])
            if count:
                pairs.append({"classes": [left, CLASSES[j]], "errors": count,
                              "a_to_b": int(matrix[i, j]), "b_to_a": int(matrix[j, i])})
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision_weighted": float(precision_score(labels, predicted, labels=list(CLASSES), average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(labels, predicted, labels=list(CLASSES), average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(labels, predicted, labels=list(CLASSES), average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, predicted, labels=list(CLASSES), average="macro", zero_division=0)),
        "per_class": report, "confusion_matrix": matrix.tolist(),
        "most_confused_pairs": sorted(pairs, key=lambda r: -r["errors"])[:5],
    }


def _save_data_figures(dataset, manifest, directory, seed):
    counts = manifest.groupby(["class", "split"]).size().unstack(fill_value=0)
    ax = counts.reindex(CLASSES)[["train", "validation", "test"]].plot.bar(figsize=(10, 4), rot=0)
    ax.set(ylabel="Nombre d'images", title="Répartition après séparation des groupes")
    ax.figure.tight_layout()
    ax.figure.savefig(directory / "class_distribution.png", dpi=130)
    plt.close(ax.figure)
    fig, axes = plt.subplots(5, 3, figsize=(9, 12))
    for i, cls in enumerate(CLASSES):
        sample = manifest[(manifest.split == "train") & (manifest["class"] == cls)].sample(3, random_state=seed)
        for ax, name in zip(axes[i], sample.image):
            ax.imshow(load_pixels(dataset / name))
            ax.set_title(cls)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(directory / "sample_images.png", dpi=110)
    plt.close(fig)


def run_training(root: Path, output_dir: Path | None = None, seed=42):
    """Point d'entrée conservé ; utilise l'unique protocole du livrable 2."""
    from backend.livrable2 import run_experiments

    return run_experiments(root, output_dir=output_dir, seed=seed)


def predict_image(artifact_path: Path, image_path: Path):
    """Recharge un artefact local de confiance et réutilise sa configuration."""
    artifact = joblib.load(artifact_path)
    pipeline = artifact["pipeline"]
    vector = extract_features(image_path, artifact["feature_config"])[None, :]
    with threadpool_limits(limits=4):
        label = str(pipeline.predict(vector)[0])
        probabilities = pipeline.predict_proba(vector)[0] if hasattr(pipeline, "predict_proba") else None
    distribution = (
        {str(cls): float(p) for cls, p in zip(pipeline.classes_, probabilities)}
        if probabilities is not None else None
    )
    model_name = artifact.get("selected_model") or artifact["configuration"]["model"]
    return {
        "image": str(image_path), "predicted_class": label,
        "confidence": distribution[label] if distribution is not None else None,
        "probabilities": distribution, "model": model_name,
        "confidence_note": (
            "Probabilité non garantie/calibrée pour la route réelle. "
            "Pour SVC, la classe prédite peut différer de l'argmax des probabilités."
            if distribution is not None else
            "Ce modèle prédit une classe sans fournir de pourcentage de confiance."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=Path, help="Inférence seule sur une image")
    args = parser.parse_args()
    if args.predict:
        output = args.output_dir or args.root / "backend" / "models" / "livrable2"
        print(json.dumps(predict_image(output / "selected_pipeline.joblib", args.predict), ensure_ascii=False, indent=2))
    else:
        run_training(args.root, args.output_dir, args.seed)


if __name__ == "__main__":
    main()
