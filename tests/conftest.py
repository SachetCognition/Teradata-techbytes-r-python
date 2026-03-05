"""
Shared pytest fixtures for the R and Python TechBytes Demo tests.

Provides sample DataFrames and mock objects for testing without
a Teradata connection.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "R_Py-Input_Tables")


@pytest.fixture
def sample_customer_df():
    """Small subset of Customer data (10 customers)."""
    return pd.DataFrame({
        'cust_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'income': [50000.0, 75000.0, 30000.0, 120000.0, 45000.0,
                   90000.0, 60000.0, 85000.0, 55000.0, 70000.0],
        'age': [25, 35, 45, 55, 30, 40, 50, 28, 38, 48],
        'years_with_bank': [5, 10, 15, 20, 3, 8, 12, 2, 7, 18],
        'nbr_children': [0, 2, 1, 3, 0, 1, 2, 0, 1, 4],
        'gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
        'marital_status': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
        'postal_code': ['90210', '10001', '60601', '85001', '43001',
                        '75001', '94102', '11201', '77001', '33101'],
        'state_code': ['CA', 'NY', 'IL', 'AZ', 'OH', 'TX', 'CA', 'NY', 'TX', 'FL'],
    })


@pytest.fixture
def sample_accounts_df():
    """Small subset of Accounts data (30 accounts for 10 customers)."""
    data = []
    acct_types = ['CK', 'SV', 'CC']
    for cust_id in range(1, 11):
        for i, acct_type in enumerate(acct_types):
            data.append({
                'acct_nbr': f'ACCT{cust_id:04d}{i}',
                'cust_id': cust_id,
                'acct_type': acct_type,
                'account_active': 'Y',
                'acct_start_date': '2020-01-15',
                'starting_balance': 1000.0 + cust_id * 100 + i * 50,
                'ending_balance': 1200.0 + cust_id * 100 + i * 50,
            })
    return pd.DataFrame(data)


@pytest.fixture
def sample_transactions_df():
    """Small subset of Transactions data (100 transactions)."""
    np.random.seed(42)
    data = []
    for tran_id in range(1, 101):
        cust_id = (tran_id % 10) + 1
        acct_idx = tran_id % 3
        month = (tran_id % 12) + 1
        data.append({
            'tran_id': tran_id,
            'acct_nbr': f'ACCT{cust_id:04d}{acct_idx}',
            'tran_amt': round(np.random.uniform(10, 500), 2),
            'principal_amt': round(np.random.uniform(10, 400), 2),
            'interest_amt': round(np.random.uniform(0, 50), 3),
            'new_balance': round(np.random.uniform(500, 5000), 2),
            'tran_date': f'2020-{month:02d}-15',
            'tran_time': 120000 + tran_id,
            'channel': np.random.choice(['A', 'B', 'C']),
            'tran_code': np.random.choice(['CR', 'DB', 'FE']),
        })
    return pd.DataFrame(data)


@pytest.fixture
def mock_td_context():
    """Mock teradataml context object."""
    ctx = MagicMock()
    ctx.execute = MagicMock(return_value=None)
    return ctx


@pytest.fixture
def customer_csv_path():
    """Path to Customer.csv file."""
    return os.path.join(INPUT_DIR, "Customer.csv")


@pytest.fixture
def accounts_csv_path():
    """Path to Accounts.csv file."""
    return os.path.join(INPUT_DIR, "Accounts.csv")
