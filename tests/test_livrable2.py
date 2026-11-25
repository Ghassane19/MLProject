"""Contrats de séparation, sélection et export du deuxième livrable sur images réelles."""

from pathlib import Path
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from backend.ml_pipeline import CLASSES, FEATURE_CONFIG, evaluate_predictions, extract_features, predict_image
from backend.livrable2 import (
    MODELS, group_bootstrap, grouped_learning_subsets,
    make_pipeline, selected_indices, training_sample,
)


class Livrable2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dataset = Path(__file__).resolve().parents[1] / "dataset"
        rows = pd.read_csv(dataset / "sources.csv")
        train = rows[rows.split.eq("train")].groupby("class").head(4)
        val = rows[rows.split.eq("test")].groupby("class").head(1)
        cls.x = np.stack([extract_features(dataset / name) for name in train.image])
        cls.y = train["class"].to_numpy()
        cls.x_val = np.stack([extract_features(dataset / name) for name in val.image])
        cls.training_rows = train.reset_index(drop=True)
        cls.development = rows[rows.split.eq("train")].reset_index(drop=True)
        cls.development["evaluation_group"] = cls.development["source_group"]

    def test_balancing_duplicates_training_rows_only_and_is_reproducible(self):
        x, y = self.x[:-2], self.y[:-2]
        for model in ("kNN", "GaussianNB", "HistGradientBoosting", "MLP"):
            with self.subTest(model=model):
                fit_x, fit_y = training_sample(x, y, model, "Correction_train")
                self.assertEqual(1, len(set(pd.Series(fit_y).value_counts())))
                for vector, label in zip(fit_x, fit_y):
                    self.assertTrue(any(np.array_equal(vector, source) for source in x[y == label]))
                repeated_x, repeated_y = training_sample(x, y, model, "Correction_train")
                np.testing.assert_array_equal(fit_x, repeated_x)
                np.testing.assert_array_equal(fit_y, repeated_y)
        np.testing.assert_array_equal(self.x[:-2], x)
        for balancing in ("Sans_correction", "Correction_train"):
            pipeline = make_pipeline("Toutes", "HistGradientBoosting", balancing)
            self.assertIsNone(pipeline.named_steps["model"].class_weight)

    def test_all_ten_families_fit_real_features_and_survive_export(self):
        self.assertEqual(10, len(MODELS))
        for model in MODELS:
            with self.subTest(model=model), threadpool_limits(limits=2):
                parameters = {"max_iter": 20, "batch_size": len(self.y)} if model == "MLP" else None
                pipeline = make_pipeline("ANOVA_50pct", model, "Correction_train", parameters=parameters)
                fit_x, fit_y = training_sample(self.x, self.y, model, "Correction_train")
                # Le petit budget MLP de ce test d'export peut arrêter avant convergence.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    pipeline.fit(fit_x, fit_y)
                predicted = pipeline.predict(self.x_val)
                self.assertEqual((len(self.x_val),), predicted.shape)
                self.assertTrue(set(predicted).issubset(CLASSES))
                self.assertTrue(np.isfinite(pipeline[:-1].transform(self.x_val)).all())
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "pipeline.joblib"
                    joblib.dump(pipeline, path)
                    restored = joblib.load(path)
                    np.testing.assert_array_equal(predicted, restored.predict(self.x_val))

    def test_hsv_selected_indices_still_refer_to_original_hog_hsv_vector(self):
        for selection in ("Toutes", "ANOVA_50pct", "Importance_RF_50pct"):
            with self.subTest(selection=selection), threadpool_limits(limits=2):
                pipeline = make_pipeline(selection, "kNN", "Sans_correction", descriptor="HSV")
                pipeline.fit(self.x, self.y)
                indices = selected_indices(pipeline)
                self.assertGreater(len(indices), 0)
                self.assertTrue(np.all((indices >= 1764) & (indices < 1860)))
                self.assertEqual(len(indices), pipeline[:-1].transform(self.x_val).shape[1])

    def test_group_bootstrap_is_reproducible_and_keeps_perfect_accuracy(self):
        groups = self.training_rows["source_group"].to_numpy()
        first = group_bootstrap(self.y, self.y.copy(), groups, n_resamples=100, seed=42)
        repeated = group_bootstrap(self.y, self.y.copy(), groups, n_resamples=100, seed=42)
        self.assertEqual(first, repeated)
        rows = {row["metric"]: row for row in first}
        self.assertEqual({"accuracy", "f1_macro"}, set(rows))
        self.assertEqual(len(np.unique(groups)), rows["accuracy"]["bootstrap_groups"])
        for key in ("estimate", "lower_95", "upper_95"):
            self.assertAlmostEqual(1.0, rows["accuracy"][key])
        self.assertAlmostEqual(1.0, rows["f1_macro"]["estimate"])
        self.assertGreaterEqual(rows["f1_macro"]["absent_class_fraction"], 0)
        self.assertLessEqual(rows["f1_macro"]["absent_class_fraction"], 1)

    def test_one_bootstrap_cluster_is_never_broken_into_individual_rows(self):
        predicted = self.y.copy()
        predicted[0] = next(label for label in CLASSES if label != self.y[0])
        # Un cluster contenant plusieurs classes représente une unité indivisible.
        # Les étiquettes et les prédictions proviennent des exemples réels ci-dessus.
        groups = np.repeat("single_shared_capture", len(self.y))
        rows = group_bootstrap(self.y, predicted, groups, n_resamples=100, seed=42)
        expected = evaluate_predictions(self.y, predicted)
        for row in rows:
            self.assertEqual(1, row["bootstrap_groups"])
            for key in ("estimate", "lower_95", "upper_95"):
                self.assertAlmostEqual(expected[row["metric"]], row[key])

    def test_learning_subsets_are_nested_reproducible_and_preserve_whole_groups(self):
        subsets = list(grouped_learning_subsets(self.development, seed=42))
        repeated = list(grouped_learning_subsets(self.development, seed=42))
        self.assertEqual(4, len(subsets))
        previous = set()
        groups = self.development["evaluation_group"]
        for (fraction, indices), (again_fraction, again_indices) in zip(subsets, repeated):
            self.assertEqual(fraction, again_fraction)
            np.testing.assert_array_equal(indices, again_indices)
            selected = set(indices)
            self.assertTrue(previous.issubset(selected))
            self.assertEqual(len(indices), len(selected))
            records = self.development.iloc[indices]
            self.assertEqual(set(CLASSES), set(records["class"]))
            selected_groups = set(records["evaluation_group"])
            expected_indices = set(np.flatnonzero(groups.isin(selected_groups).to_numpy()))
            self.assertEqual(expected_indices, selected)
            previous = selected
        self.assertEqual(1.0, subsets[-1][0])
        self.assertEqual(set(range(len(self.development))), previous)

    def test_inference_leaves_training_statistics_and_selected_variables_unchanged(self):
        for selection in ("Toutes", "ANOVA_50pct", "Importance_RF_50pct"):
            with self.subTest(selection=selection), threadpool_limits(limits=2):
                pipeline = make_pipeline(selection, "kNN", "Sans_correction").fit(self.x, self.y)
                indices = selected_indices(pipeline).copy()
                mean = pipeline.named_steps["scaler"].mean_.copy()
                pipeline.predict(self.x_val)
                np.testing.assert_array_equal(indices, selected_indices(pipeline))
                np.testing.assert_array_equal(mean, pipeline.named_steps["scaler"].mean_)
                retained = pipeline.named_steps["variance"].transform(self.x)
                np.testing.assert_allclose(mean, retained.mean(axis=0), atol=1e-7)
                self.assertEqual(len(indices), pipeline[:-1].transform(self.x_val).shape[1])
                if selection != "Toutes":
                    self.assertLess(len(indices), retained.shape[1])

    def test_model_based_selector_survives_export_and_keeps_predictions(self):
        with threadpool_limits(limits=2):
            pipeline = make_pipeline("Importance_RF_50pct", "SVM", "Correction_train").fit(self.x, self.y)
            expected = pipeline.predict(self.x_val)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "pipeline.joblib"
                joblib.dump(pipeline, path)
                restored = joblib.load(path)
                np.testing.assert_array_equal(expected, restored.predict(self.x_val))
                np.testing.assert_array_equal(selected_indices(pipeline), selected_indices(restored))

    def test_current_svm_inference_returns_class_without_inventing_probabilities(self):
        dataset = Path(__file__).resolve().parents[1] / "dataset"
        rows = pd.read_csv(dataset / "sources.csv")
        image_path = dataset / rows[rows.split.eq("test")].iloc[0].image
        with threadpool_limits(limits=2), tempfile.TemporaryDirectory() as directory:
            pipeline = make_pipeline("Toutes", "SVM", "Sans_correction").fit(self.x, self.y)
            path = Path(directory) / "selected_pipeline.joblib"
            joblib.dump({"pipeline": pipeline, "feature_config": FEATURE_CONFIG,
                         "configuration": {"model": "SVM"}}, path)
            prediction = predict_image(path, image_path)
            expected = pipeline.predict(extract_features(image_path)[None, :])[0]
        self.assertEqual(expected, prediction["predicted_class"])
        self.assertEqual("SVM", prediction["model"])
        self.assertIsNone(prediction["confidence"])
        self.assertIsNone(prediction["probabilities"])


if __name__ == "__main__":
    unittest.main()
