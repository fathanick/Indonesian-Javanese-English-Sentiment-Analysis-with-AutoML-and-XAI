"""Offline acceptance tests, intentionally not run during code preparation."""
import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from ije.data import clean_text, grouped_folds
from ije.features import TextFeatures, cm_row
from ije.metrics import metrics
from ije.explain import aggregate_weights
from ije.storage import ROOT, authorize, read_json, write_json, require_stage, finish_stage
from ije.validation import validate_protocol


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.config = read_json(ROOT / "config" / "protocol.json")

    def test_preprocessing_preserves_sentiment_content(self):
        self.assertEqual(clean_text("@user TIDAK baguuus!!! 😊 #Good 123 https://x.test"),
                         "tidak baguuus!!! 😊 good 123")
        self.assertEqual(clean_text(clean_text("#Apik @a")), "apik")

    def test_other_tags_are_not_another_language_for_single_cmi(self):
        result = cm_row(["ID", "ID", "OTH", "OTH"])
        self.assertEqual(result["cmi_single_only"], 0)
        self.assertEqual(result["tag_diversity_all"], .5)
        self.assertEqual(len(result), 21)

    def test_switch_rate_uses_retained_transitions(self):
        result = cm_row(["ID", "OTH", "JV"])
        self.assertEqual(result["switch_count"], 1)
        self.assertEqual(result["switch_rate"], 1)

    def test_empty_cm_and_unknown_labels(self):
        self.assertTrue(all(v == 0 for v in cm_row([]).values()))
        with self.assertRaises(ValueError):
            cm_row(["UNKNOWN"])

    def test_validation_text_cannot_change_fitted_vocabulary_or_scaler(self):
        f = TextFeatures("tfidf", self.config)
        f.fit_transform(["apik bagus layanan", "elek buruk layanan", "biasa info layanan"] * 3)
        vocab = dict(f.word.vocabulary_)
        mean = f.scalers["stats"].mean_.copy()
        f.transform(["unseenvalidationtoken " * 100])
        self.assertNotIn("unseenvalidationtoken", f.word.vocabulary_)
        self.assertEqual(vocab, f.word.vocabulary_)
        np.testing.assert_array_equal(mean, f.scalers["stats"].mean_)

    def test_grouped_folds_keep_duplicates_together(self):
        df = pd.DataFrame({"label_id": [i % 3 for i in range(90)], "group": [f"g{i}" for i in range(90)]})
        df = pd.concat([df, df.iloc[:6]], ignore_index=True)
        folds = grouped_folds(df, 5, 42)
        covered = []
        for train, valid in folds:
            self.assertFalse(set(df.iloc[train].group) & set(df.iloc[valid].group))
            covered.extend(valid)
        self.assertEqual(sorted(covered), list(range(len(df))))

    def test_accuracy_and_macro_f1_are_distinct(self):
        result = metrics([0, 0, 0, 1, 2], [0, 0, 0, 0, 0])
        self.assertAlmostEqual(result["accuracy"], .6)
        self.assertAlmostEqual(result["f1_macro"], .25)

    def test_occurrence_counts_are_not_word_counts(self):
        rows = [{"target_class": "positive", "word": "good", "language": "EN",
                 "sample_id": f"s{i}", "weight": w} for i, w in enumerate([.2, .1, -.1])]
        words, languages = aggregate_weights(rows, 3)
        self.assertEqual(len(words), 1)
        self.assertEqual(languages[0]["n_distinct_words"], 1)
        self.assertEqual(languages[0]["n_attributions"], 3)
        self.assertEqual(languages[0]["n_positive_attributions"], 2)

    def test_approval_default_denies_execution(self):
        c = copy.deepcopy(self.config)
        c["execution_enabled"] = False
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.json"
            write_json(path, c)
            with self.assertRaises(RuntimeError):
                authorize(path, Path(d) / "missing.json", "develop")

    def test_changed_stage_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "result.txt"
            file.write_text("original")
            finish_stage(d, "probe", [file])
            require_stage(d, "probe")
            file.write_text("modified")
            with self.assertRaises(RuntimeError):
                require_stage(d, "probe")

    def test_primary_metric_cannot_silently_change(self):
        c = copy.deepcopy(self.config)
        c["primary_metric"] = "accuracy"
        with self.assertRaises(ValueError):
            validate_protocol(c)

    def test_selection_ignores_opposite_test_ranking(self):
        from ije.experiment import select_development
        rows = [{"scenario_id": "development_winner", "mean_outer_macro_f1": .92, "test_f1": .88},
                {"scenario_id": "test_winner", "mean_outer_macro_f1": .90, "test_f1": .95}]
        self.assertEqual(select_development(rows)["scenario_id"], "development_winner")

    def test_selection_ties_follow_prespecified_order(self):
        from ije.experiment import select_development
        rows = [{"scenario_id": "first", "mean_outer_macro_f1": .90},
                {"scenario_id": "second", "mean_outer_macro_f1": .90}]
        self.assertEqual(select_development(rows)["scenario_id"], "first")


if __name__ == "__main__":
    unittest.main()
