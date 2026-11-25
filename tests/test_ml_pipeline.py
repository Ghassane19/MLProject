"""Checks of data independence and the saved inference contract on real images.

Run from the repository root with ``python -m unittest discover -s tests -v``.
All generated fixtures and models live in temporary directories.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier

from backend.ml_pipeline import (
    CLASSES, FEATURE_CONFIG, _pipeline, evaluate_predictions, extract_features, load_pixels,
    predict_image, prepare_manifest, split_development,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"


class RealDataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(DATASET / "sources.csv", keep_default_na=False)

    def temporary_dataset(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        dataset = Path(directory.name) / "dataset"
        dataset.mkdir()
        for row in rows.to_dict("records"):
            target = dataset / row["image"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(DATASET / row["image"], target)
        rows.to_csv(dataset / "sources.csv", index=False)
        return dataset


class DevelopmentSplitTests(RealDataTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.frame = cls.metadata.copy()
        cls.frame["evaluation_group"] = cls.frame.source_group
        cls.selected, cls.summary = split_development(cls.frame)

    def test_groups_remain_indivisible_and_test_is_unchanged(self):
        self.assertTrue(self.selected.groupby("evaluation_group").split.nunique().eq(1).all())
        original_test = set(self.frame.loc[self.frame.split == "test", "image"])
        selected_test = set(self.selected.loc[self.selected.split == "test", "image"])
        self.assertEqual(original_test, selected_test)
        self.assertEqual(len(self.frame), len(self.selected))
        for split in ("train", "validation", "test"):
            self.assertEqual(set(CLASSES), set(self.selected.loc[self.selected.split == split, "class"]))

    def test_large_pedestrian_group_does_not_create_tiny_validation(self):
        # Real provenance has only 531-, 256- and 13-person training groups.
        pedestrians = self.selected[self.selected["class"] == "pedestrian"]
        self.assertEqual(256, int(pedestrians.split.eq("validation").sum()))
        self.assertEqual(544, int(pedestrians.split.eq("train").sum()))
        self.assertEqual(200, int(pedestrians.split.eq("test").sum()))
        self.assertEqual(0, self.summary["group_overlap"])

    def test_partition_is_reproducible(self):
        repeated, summary = split_development(self.frame, seed=42)
        pd.testing.assert_frame_equal(self.selected, repeated)
        self.assertEqual(self.summary, summary)

    def test_existing_validation_rejects_shared_source_groups(self):
        frame = self.frame.copy()
        index = frame.index[(frame.split == "train") & frame.source_group.eq("group_00567")][0]
        frame.loc[index, "split"] = "validation"
        with self.assertRaisesRegex(ValueError, "Fuite de groupes"):
            split_development(frame)


class ManifestTests(RealDataTestCase):
    def test_identical_pixels_across_splits_are_kept_only_in_test(self):
        rows = self.metadata[(self.metadata["class"] == "bus") & self.metadata.split.eq("train")].head(1).copy()
        dataset = self.temporary_dataset(rows)
        original = dataset / rows.iloc[0].image
        duplicate = rows.iloc[0].to_dict()
        duplicate.update(image="test/bus/reencoded.png", split="test", source_group="test_duplicate_group")
        target = dataset / duplicate["image"]
        target.parent.mkdir(parents=True)
        with Image.open(original) as image:
            image.save(target, compress_level=0)
        self.assertNotEqual(original.read_bytes(), target.read_bytes())
        pd.concat([rows, pd.DataFrame([duplicate])], ignore_index=True).to_csv(dataset / "sources.csv", index=False)
        before = hashlib.sha256(original.read_bytes()).hexdigest()

        manifest, audit = prepare_manifest(dataset)

        self.assertEqual([duplicate["image"]], manifest.image.tolist())
        self.assertEqual("doublon exact", audit["excluded"][0]["reason"])
        self.assertEqual(rows.iloc[0].image, audit["excluded"][0]["image"])
        self.assertTrue(original.exists())
        self.assertEqual(before, hashlib.sha256(original.read_bytes()).hexdigest())

    def test_changed_pixels_cannot_silently_use_old_provenance(self):
        rows = self.metadata[self.metadata.split.eq("train")].head(1).copy()
        dataset = self.temporary_dataset(rows)
        other = self.metadata[self.metadata.split.eq("train")].iloc[1]
        shutil.copyfile(DATASET / other.image, dataset / rows.iloc[0].image)
        with self.assertRaisesRegex(ValueError, "Pixels différents"):
            prepare_manifest(dataset)

    def test_contradictory_labels_on_identical_pixels_require_review(self):
        rows = self.metadata[self.metadata.split.eq("train")].head(1).copy()
        dataset = self.temporary_dataset(rows)
        duplicate = rows.iloc[0].to_dict()
        duplicate.update(image="test/car/conflicting.png", split="test", **{"class": "car"})
        target = dataset / duplicate["image"]
        target.parent.mkdir(parents=True)
        shutil.copyfile(dataset / rows.iloc[0].image, target)
        pd.concat([rows, pd.DataFrame([duplicate])], ignore_index=True).to_csv(dataset / "sources.csv", index=False)
        with self.assertRaisesRegex(ValueError, "étiquettes contradictoires"):
            prepare_manifest(dataset)


class FeatureAndInferenceTests(RealDataTestCase):
    def test_single_class_source_metrics_keep_the_global_five_class_definition(self):
        labels = self.metadata.loc[self.metadata.source.eq("penn_fudan"), "class"].head(3).to_numpy()
        scores = evaluate_predictions(labels, labels.copy())
        self.assertAlmostEqual(1.0, scores["accuracy"])
        self.assertAlmostEqual(1.0, scores["f1_weighted"])
        self.assertAlmostEqual(1 / len(CLASSES), scores["f1_macro"])
        self.assertAlmostEqual(scores["per_class"]["macro avg"]["f1-score"], scores["f1_macro"])
        self.assertEqual((5, 5), np.asarray(scores["confusion_matrix"]).shape)

    def test_features_are_independent_of_image_file_encoding(self):
        image_path = DATASET / self.metadata[self.metadata.split.eq("train")].iloc[0].image
        with tempfile.TemporaryDirectory() as directory:
            reencoded = Path(directory) / "same_pixels.bmp"
            with Image.open(image_path) as image:
                image.convert("RGB").save(reencoded)
            original = extract_features(image_path)
            np.testing.assert_allclose(original, extract_features(reencoded), atol=0, rtol=0)
        self.assertEqual((1860,), original.shape)
        self.assertEqual(np.float32, original.dtype)
        self.assertTrue(np.isfinite(original).all())
        for histogram in original[-96:].reshape(3, 32):
            self.assertAlmostEqual(1.0, float(histogram.sum()), places=6)

    def test_saved_inference_reuses_feature_configuration_and_class_probabilities(self):
        config = deepcopy(FEATURE_CONFIG)
        config["hsv_bins_per_channel"] = 8
        training = self.metadata[self.metadata.split.eq("train")].groupby("class", sort=False).head(2)
        paths = [DATASET / image for image in training.image]
        vectors = np.stack([extract_features(path, config) for path in paths])
        labels = training["class"].to_numpy()
        pipeline = _pipeline(KNeighborsClassifier(n_neighbors=1, weights="distance"))
        pipeline.fit(vectors, labels)
        np.testing.assert_allclose(pipeline.named_steps["scaler"].mean_, vectors.mean(axis=0), atol=1e-7)

        # Evaluation and export must leave the fitted training statistics unchanged.
        image_path = DATASET / self.metadata[self.metadata.split.eq("test")].iloc[0].image
        fitted_mean = pipeline.named_steps["scaler"].mean_.copy()
        vector = extract_features(image_path, config)[None, :]
        expected_label = str(pipeline.predict(vector)[0])
        expected_distribution = dict(zip(pipeline.classes_, pipeline.predict_proba(vector)[0]))
        artifact = {
            "pipeline": pipeline,
            "scaler": pipeline.named_steps["scaler"],
            "best_model": pipeline.named_steps["model"],
            "selected_model": "kNN",
            "feature_config": config,
            "class_names": list(CLASSES),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classical_pipeline.joblib"
            joblib.dump(artifact, path)
            prediction = predict_image(path, image_path)
            restored = joblib.load(path)

        self.assertEqual(expected_label, prediction["predicted_class"])
        self.assertEqual(expected_distribution, prediction["probabilities"])
        self.assertEqual(expected_distribution[expected_label], prediction["confidence"])
        self.assertEqual(set(CLASSES), set(prediction["probabilities"]))
        self.assertAlmostEqual(1.0, sum(prediction["probabilities"].values()))
        self.assertEqual("kNN", prediction["model"])
        np.testing.assert_array_equal(fitted_mean, restored["scaler"].mean_)
        self.assertEqual((1, 1788), vector.shape)

    def test_pixels_follow_normalized_rgb_contract(self):
        path = DATASET / self.metadata[self.metadata.split.eq("train")].iloc[0].image
        pixels = load_pixels(path)
        self.assertEqual((128, 128, 3), pixels.shape)
        self.assertEqual(np.float32, pixels.dtype)
        self.assertTrue(np.isfinite(pixels).all())
        self.assertGreaterEqual(float(pixels.min()), 0)
        self.assertLessEqual(float(pixels.max()), 1)


if __name__ == "__main__":
    unittest.main()
