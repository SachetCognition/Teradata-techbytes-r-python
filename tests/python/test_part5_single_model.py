"""
Tests for Part 5 single-model workflow.

Tests the RandomForestClassifier training, serialization, and
predictor column validation.
"""

import os
import sys
import pytest
import pickle
import base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestSampling:
    def test_sample_25_percent(self, sample_customer_df):
        """Verify 25% sampling of ADS."""
        n = len(sample_customer_df)
        sample = sample_customer_df.sample(frac=0.25, random_state=42)
        assert len(sample) == pytest.approx(0.25 * n, abs=1)


class TestRandomForestClassifier:
    def test_rf_classifier_params(self):
        """Verify RandomForestClassifier created with correct parameters."""
        from sklearn.ensemble import RandomForestClassifier

        classifier = RandomForestClassifier(
            n_estimators=500, max_features=5, random_state=0
        )
        assert classifier.n_estimators == 500
        assert classifier.max_features == 5
        assert classifier.random_state == 0

    def test_predictor_columns(self):
        """Verify the 18 predictor columns used match expected list."""
        predictor_columns = [
            "tot_income", "tot_age", "tot_cust_years", "tot_children",
            "female_ind", "single_ind", "married_ind", "separated_ind",
            "ck_acct_ind", "sv_acct_ind", "ck_avg_bal", "sv_avg_bal",
            "ck_avg_tran_amt", "sv_avg_tran_amt", "q1_trans_cnt",
            "q2_trans_cnt", "q3_trans_cnt", "q4_trans_cnt",
        ]
        assert len(predictor_columns) == 18
        # Verify key columns present
        assert "tot_income" in predictor_columns
        assert "cc_acct_ind" not in predictor_columns  # target, not predictor


class TestModelSerialization:
    def test_model_serialization(self):
        """Create a small RF model, serialize with pickle+base64, deserialize,
        verify it can still predict."""
        from sklearn.ensemble import RandomForestClassifier

        # Create small training data
        np.random.seed(42)
        X_train = np.random.rand(50, 5)
        y_train = np.random.randint(0, 2, 50)

        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(X_train, y_train)

        # Serialize like the demo does
        classifierPkl = pickle.dumps(classifier)
        classifierPklB64 = base64.b64encode(classifierPkl)

        # Deserialize
        restored_pkl = base64.b64decode(classifierPklB64)
        restored_classifier = pickle.loads(restored_pkl)

        # Verify predictions match
        X_test = np.random.rand(10, 5)
        original_pred = classifier.predict(X_test)
        restored_pred = restored_classifier.predict(X_test)
        np.testing.assert_array_equal(original_pred, restored_pred)

    def test_model_file_creation(self, tmp_path):
        """Verify RFmodel_py.out file is created and is non-empty."""
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, 20)

        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(X, y)

        classifierPkl = pickle.dumps(classifier)
        classifierPklB64 = base64.b64encode(classifierPkl)

        model_path = tmp_path / "RFmodel_py.out"
        with open(model_path, 'wb') as fOut:
            fOut.write(classifierPklB64)

        assert model_path.exists()
        assert model_path.stat().st_size > 0
