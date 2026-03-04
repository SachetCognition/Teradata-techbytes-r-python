"""
Tests for data loading: CSV file validation.

Tests verify that the source CSV files are readable, have expected columns,
correct dtypes, unique primary keys, and valid domain values.
"""

import os
import pytest
import pandas as pd

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "R_Py-Input_Tables")


@pytest.fixture
def customer_df():
    return pd.read_csv(os.path.join(INPUT_DIR, "Customer.csv"))


@pytest.fixture
def accounts_df():
    return pd.read_csv(os.path.join(INPUT_DIR, "Accounts.csv"))


class TestCustomerCSV:
    def test_customer_csv_readable(self, customer_df):
        """Verify Customer.csv can be read by pandas with expected columns."""
        expected_columns = [
            'cust_id', 'income', 'age', 'years_with_bank', 'nbr_children',
            'gender', 'marital_status', 'postal_code', 'state_code'
        ]
        for col in expected_columns:
            assert col in customer_df.columns, f"Missing column: {col}"

    def test_customer_csv_no_nulls_in_pk(self, customer_df):
        """Verify cust_id has no nulls."""
        assert customer_df['cust_id'].notna().all(), "cust_id has null values"

    def test_customer_csv_unique_pk(self, customer_df):
        """Verify cust_id values are unique."""
        assert customer_df['cust_id'].is_unique, "cust_id has duplicate values"

    def test_csv_row_counts_customer(self, customer_df):
        """Verify Customer CSV has approximately 10K rows."""
        assert len(customer_df) >= 9000, f"Customer has only {len(customer_df)} rows"
        assert len(customer_df) <= 11000, f"Customer has {len(customer_df)} rows (expected ~10K)"

    def test_gender_values(self, customer_df):
        """Verify gender column only contains expected values (M, F)."""
        valid_genders = {'M', 'F'}
        actual_genders = set(customer_df['gender'].dropna().unique())
        assert actual_genders.issubset(valid_genders), \
            f"Unexpected gender values: {actual_genders - valid_genders}"

    def test_marital_status_values(self, customer_df):
        """Verify marital_status only contains expected values (1, 2, 3, 4)."""
        valid_statuses = {1, 2, 3, 4}
        actual_statuses = set(int(x) for x in customer_df['marital_status'].dropna().unique())
        assert actual_statuses.issubset(valid_statuses), \
            f"Unexpected marital_status values: {actual_statuses - valid_statuses}"

    def test_state_code_values(self, customer_df):
        """Verify state_code contains valid US state/province codes."""
        # Include Canadian province codes that may appear in sample data
        valid_states = {
            'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
            'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
            'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
            'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
            'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
            'DC', 'NB', 'AB', 'BC', 'MB', 'ON', 'QC', 'SK',
        }
        actual_states = set(customer_df['state_code'].dropna().str.strip().unique())
        assert actual_states.issubset(valid_states), \
            f"Unexpected state_code values: {actual_states - valid_states}"


class TestAccountsCSV:
    def test_accounts_csv_readable(self, accounts_df):
        """Verify Accounts.csv has expected columns."""
        expected_columns = [
            'acct_nbr', 'cust_id', 'acct_type', 'account_active',
            'acct_start_date', 'starting_balance', 'ending_balance'
        ]
        for col in expected_columns:
            assert col in accounts_df.columns, f"Missing column: {col}"

    def test_accounts_csv_no_nulls_in_pk(self, accounts_df):
        """Verify acct_nbr has no nulls."""
        assert accounts_df['acct_nbr'].notna().all(), "acct_nbr has null values"

    def test_accounts_csv_unique_pk(self, accounts_df):
        """Verify acct_nbr values are unique."""
        assert accounts_df['acct_nbr'].is_unique, "acct_nbr has duplicate values"

    def test_csv_row_counts_accounts(self, accounts_df):
        """Verify Accounts CSV has a reasonable number of rows."""
        assert len(accounts_df) >= 10000, f"Accounts has only {len(accounts_df)} rows"
        assert len(accounts_df) <= 200000, f"Accounts has {len(accounts_df)} rows (unexpectedly large)"

    def test_acct_type_values(self, accounts_df):
        """Verify acct_type only contains expected values (CK, SV, CC)."""
        valid_types = {'CK', 'SV', 'CC'}
        actual_types = set(accounts_df['acct_type'].dropna().str.strip().unique())
        assert actual_types.issubset(valid_types), \
            f"Unexpected acct_type values: {actual_types - valid_types}"
