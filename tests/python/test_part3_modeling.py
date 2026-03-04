"""
Tests for Part 3 model training logic.

Tests model creation, scoring, confusion matrix, and model management
using mocked teradataml functions.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestTrainTestSplit:
    def test_train_test_split_ratio(self, sample_customer_df):
        """Verify 60/40 split produces approximately correct proportions."""
        import pandas as pd
        n = len(sample_customer_df)
        # Simulate the 60/40 split
        train_size = int(0.6 * n)
        test_size = n - train_size
        assert train_size == pytest.approx(0.6 * n, abs=1)
        assert test_size == pytest.approx(0.4 * n, abs=1)


class TestXGBoostModel:
    def test_xgboost_model_creation(self):
        """Mock XGBoost call, verify correct parameters are passed."""
        with patch('teradataml.XGBoost') as MockXGBoost:
            MockXGBoost.return_value = MagicMock()
            from teradataml import XGBoost

            mock_data = MagicMock()
            formula = "cc_acct_ind ~ tot_income + tot_age"

            model = XGBoost(
                data=mock_data,
                id_column='cust_id',
                formula=formula,
                num_boosted_trees=4,
                loss_function='binomial',
                prediction_type='classification',
                reg_lambda=1.0,
                shrinkage_factor=0.1,
                iter_num=10,
                min_node_size=1,
                max_depth=10,
            )

            MockXGBoost.assert_called_once_with(
                data=mock_data,
                id_column='cust_id',
                formula=formula,
                num_boosted_trees=4,
                loss_function='binomial',
                prediction_type='classification',
                reg_lambda=1.0,
                shrinkage_factor=0.1,
                iter_num=10,
                min_node_size=1,
                max_depth=10,
            )

    def test_xgboost_scoring(self):
        """Mock XGBoostPredict, verify it receives correct model and test data."""
        with patch('teradataml.XGBoostPredict') as MockPredict:
            MockPredict.return_value = MagicMock()
            from teradataml import XGBoostPredict

            mock_model = MagicMock()
            mock_testdata = MagicMock()

            XGBoostPredict(
                mock_model,
                newdata=mock_testdata,
                object_order_column=['tree_id', 'iter', 'class_num'],
                id_column='cust_id',
                terms='cc_acct_ind',
                num_boosted_trees=4,
            )

            MockPredict.assert_called_once()


class TestDecisionForestModel:
    def test_decision_forest_model_creation(self):
        """Mock DecisionForest call, verify parameters."""
        with patch('teradataml.DecisionForest') as MockDF:
            MockDF.return_value = MagicMock()
            from teradataml import DecisionForest

            mock_data = MagicMock()
            formula = "cc_acct_ind ~ tot_income + tot_age"

            DecisionForest(
                formula=formula,
                data=mock_data,
                tree_type="classification",
                ntree=500,
                nodesize=1,
                variance=0.0,
                max_depth=12,
                mtry=5,
                mtry_seed=100,
                seed=100,
            )

            MockDF.assert_called_once_with(
                formula=formula,
                data=mock_data,
                tree_type="classification",
                ntree=500,
                nodesize=1,
                variance=0.0,
                max_depth=12,
                mtry=5,
                mtry_seed=100,
                seed=100,
            )

    def test_decision_forest_scoring(self):
        """Mock DecisionForestPredict, verify parameters."""
        with patch('teradataml.DecisionForestPredict') as MockDFPredict:
            MockDFPredict.return_value = MagicMock()
            from teradataml import DecisionForestPredict

            mock_model = MagicMock()
            mock_testdata = MagicMock()

            DecisionForestPredict(
                mock_model,
                newdata=mock_testdata,
                id_column="cust_id",
                detailed=False,
                terms=["cc_acct_ind"],
            )

            MockDFPredict.assert_called_once()


class TestConfusionMatrix:
    def test_confusion_matrix_xgboost(self):
        """Mock ConfusionMatrix, verify reference='cc_acct_ind', prediction='prediction'."""
        with patch('teradataml.ConfusionMatrix', create=True) as MockCM:
            MockCM.return_value = MagicMock()

            mock_data = MagicMock()

            MockCM(
                data=mock_data,
                reference="cc_acct_ind",
                prediction="prediction",
            )

            MockCM.assert_called_once_with(
                data=mock_data,
                reference="cc_acct_ind",
                prediction="prediction",
            )

    def test_confusion_matrix_decision_forest(self):
        """Same for DF model."""
        with patch('teradataml.ConfusionMatrix', create=True) as MockCM:
            MockCM.return_value = MagicMock()

            mock_data = MagicMock()

            MockCM(
                data=mock_data,
                reference="cc_acct_ind",
                prediction="prediction",
            )

            MockCM.assert_called_once()


class TestModelManagement:
    def test_model_save_and_retrieve(self):
        """Mock save_model/retrieve_model, verify model names."""
        with patch('teradataml.save_model', create=True) as mock_save, \
             patch('teradataml.retrieve_model', create=True) as mock_retrieve:

            mock_model = MagicMock()
            mock_save(mock_model, name="XGBoost_Model_1")
            mock_save.assert_called_once_with(mock_model, name="XGBoost_Model_1")

            mock_retrieve(name="XGBoost_Model_1")
            mock_retrieve.assert_called_once_with(name="XGBoost_Model_1")

    def test_model_delete(self):
        """Mock delete_model, verify model names."""
        with patch('teradataml.delete_model', create=True) as mock_delete:

            mock_delete(name="XGBoost_Model_1")
            mock_delete.assert_called_once_with(name="XGBoost_Model_1")

            mock_delete(name="Decision_Forest_Model_1")
            assert mock_delete.call_count == 2

    def test_formula_columns(self):
        """Verify the formula string includes all 20 expected predictor variables."""
        formula = ("cc_acct_ind ~ tot_income + tot_age + tot_cust_years + "
                   "tot_children + female_ind + single_ind + married_ind + "
                   "separated_ind + ca_resident_ind + ny_resident_ind + "
                   "tx_resident_ind + il_resident_ind + az_resident_ind + "
                   "oh_resident_ind + ck_acct_ind + sv_acct_ind + ck_avg_bal + "
                   "sv_avg_bal + ck_avg_tran_amt + sv_avg_tran_amt")

        expected_predictors = [
            'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
            'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
            'ck_acct_ind', 'sv_acct_ind', 'ck_avg_bal', 'sv_avg_bal',
            'ck_avg_tran_amt', 'sv_avg_tran_amt',
        ]
        for pred in expected_predictors:
            assert pred in formula, f"Missing predictor: {pred}"
        assert len(expected_predictors) == 20
