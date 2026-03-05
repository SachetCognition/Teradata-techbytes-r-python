"""
Tests for Part 3 feature engineering logic.

Tests the extracted feature engineering functions using pandas DataFrames
(no Teradata dependency needed).
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from R_Py_Part_3_utils import feature_engineering


class TestCustomerFeatures:
    def test_female_indicator(self, sample_customer_df):
        """Given gender='F', female_ind should be 1; gender='M' should be 0."""
        result = feature_engineering.create_customer_features(sample_customer_df)
        for _, row in result.iterrows():
            if row['gender'] == 'F':
                assert row['female_ind'] == 1, "Female should have female_ind=1"
            else:
                assert row['female_ind'] == 0, "Male should have female_ind=0"

    def test_single_indicator(self, sample_customer_df):
        """marital_status=1 -> single_ind=1, else 0."""
        result = feature_engineering.create_customer_features(sample_customer_df)
        for _, row in result.iterrows():
            expected = 1 if row['marital_status'] == 1 else 0
            assert row['single_ind'] == expected

    def test_married_indicator(self, sample_customer_df):
        """marital_status=2 -> married_ind=1, else 0."""
        result = feature_engineering.create_customer_features(sample_customer_df)
        for _, row in result.iterrows():
            expected = 1 if row['marital_status'] == 2 else 0
            assert row['married_ind'] == expected

    def test_separated_indicator(self, sample_customer_df):
        """marital_status=3 -> separated_ind=1, else 0."""
        result = feature_engineering.create_customer_features(sample_customer_df)
        for _, row in result.iterrows():
            expected = 1 if row['marital_status'] == 3 else 0
            assert row['separated_ind'] == expected

    def test_state_indicators(self, sample_customer_df):
        """state_code='CA' -> ca_resident_ind=1, all others 0; same for other states."""
        result = feature_engineering.create_customer_features(sample_customer_df)
        state_cols = {
            'CA': 'ca_resident_ind', 'NY': 'ny_resident_ind',
            'TX': 'tx_resident_ind', 'IL': 'il_resident_ind',
            'AZ': 'az_resident_ind', 'OH': 'oh_resident_ind',
        }
        for _, row in result.iterrows():
            for state, col in state_cols.items():
                expected = 1 if row['state_code'] == state else 0
                assert row[col] == expected, \
                    f"For state_code={row['state_code']}, {col} should be {expected}"


class TestAccountFeatures:
    def test_account_type_indicators(self, sample_accounts_df):
        """acct_type='CK' -> ck_acct_ind=1, sv_acct_ind=0, cc_acct_ind=0."""
        result = feature_engineering.create_account_features(sample_accounts_df)
        for _, row in result.iterrows():
            if row['acct_type'] == 'CK':
                assert row['ck_acct_ind'] == 1
                assert row['sv_acct_ind'] == 0
                assert row['cc_acct_ind'] == 0
            elif row['acct_type'] == 'SV':
                assert row['ck_acct_ind'] == 0
                assert row['sv_acct_ind'] == 1
                assert row['cc_acct_ind'] == 0
            elif row['acct_type'] == 'CC':
                assert row['ck_acct_ind'] == 0
                assert row['sv_acct_ind'] == 0
                assert row['cc_acct_ind'] == 1

    def test_avg_balance_calculation(self, sample_accounts_df):
        """Verify ck_bal = (starting_balance + ending_balance) for CK accounts, else 0."""
        result = feature_engineering.create_account_features(sample_accounts_df)
        for _, row in result.iterrows():
            expected_bal = row['starting_balance'] + row['ending_balance']
            if row['acct_type'] == 'CK':
                assert row['ck_bal'] == pytest.approx(expected_bal)
                assert row['sv_bal'] == 0
                assert row['cc_bal'] == 0
            elif row['acct_type'] == 'SV':
                assert row['ck_bal'] == 0
                assert row['sv_bal'] == pytest.approx(expected_bal)
                assert row['cc_bal'] == 0
            elif row['acct_type'] == 'CC':
                assert row['ck_bal'] == 0
                assert row['sv_bal'] == 0
                assert row['cc_bal'] == pytest.approx(expected_bal)


class TestTransactionFeatures:
    def test_quarterly_transaction_counts(self):
        """Verify q1_trans counts transactions in months 1-3, q2 in 4-6, etc."""
        df = pd.DataFrame({
            'tran_date': ['2020-01-15', '2020-04-15', '2020-07-15', '2020-10-15'],
            'acct_type': ['CK', 'CK', 'CK', 'CK'],
            'principal_amt': [100, 200, 300, 400],
            'interest_amt': [10, 20, 30, 40],
        })
        result = feature_engineering.create_transaction_features(df)
        assert result.iloc[0]['q1_trans'] == 1
        assert result.iloc[0]['q2_trans'] == 0
        assert result.iloc[1]['q2_trans'] == 1
        assert result.iloc[1]['q1_trans'] == 0
        assert result.iloc[2]['q3_trans'] == 1
        assert result.iloc[3]['q4_trans'] == 1


class TestBuildADS:
    def test_ads_columns(self, sample_customer_df, sample_accounts_df, sample_transactions_df):
        """Verify final ADS has all 28 expected columns."""
        ads = feature_engineering.build_ads(
            sample_customer_df, sample_accounts_df, sample_transactions_df
        )
        expected_columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
            'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
            'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal',
            'ck_avg_tran_amt', 'sv_avg_tran_amt', 'cc_avg_tran_amt',
            'q1_trans_cnt', 'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
        ]
        for col in expected_columns:
            assert col in ads.columns, f"Missing ADS column: {col}"

    def test_ads_aggregation_by_cust_id(self, sample_customer_df, sample_accounts_df, sample_transactions_df):
        """Verify ADS is grouped by cust_id with one row per customer."""
        ads = feature_engineering.build_ads(
            sample_customer_df, sample_accounts_df, sample_transactions_df
        )
        assert ads['cust_id'].is_unique, "ADS should have one row per cust_id"
        assert len(ads) == len(sample_customer_df), \
            "ADS should have same number of rows as customers"

    def test_null_handling(self, sample_customer_df, sample_accounts_df, sample_transactions_df):
        """Verify None values are replaced with 0 for indicator and amount columns."""
        ads = feature_engineering.build_ads(
            sample_customer_df, sample_accounts_df, sample_transactions_df
        )
        # Check no nulls in indicator/amount columns
        indicator_cols = [
            'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal',
        ]
        for col in indicator_cols:
            assert ads[col].notna().all(), f"Column {col} has null values"

    def test_join_correctness(self, sample_customer_df):
        """Verify left joins don't lose customers who have no accounts."""
        # Create a customer with no matching accounts
        empty_accounts = pd.DataFrame({
            'acct_nbr': ['ACCT9999'],
            'cust_id': [9999],
            'acct_type': ['CK'],
            'account_active': ['Y'],
            'acct_start_date': ['2020-01-15'],
            'starting_balance': [100.0],
            'ending_balance': [200.0],
        })
        empty_transactions = pd.DataFrame({
            'tran_id': [1],
            'acct_nbr': ['ACCT9999'],
            'tran_amt': [50.0],
            'principal_amt': [40.0],
            'interest_amt': [10.0],
            'new_balance': [200.0],
            'tran_date': ['2020-01-15'],
            'tran_time': [120000],
            'channel': ['A'],
            'tran_code': ['CR'],
        })
        ads = feature_engineering.build_ads(
            sample_customer_df, empty_accounts, empty_transactions
        )
        # All customers should still be present
        assert len(ads) == len(sample_customer_df)
