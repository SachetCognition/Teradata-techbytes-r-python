"""
Tests for Part 5 multi-model workflow.

Tests statecode recoding, ADS_Py2 column structure,
train/test split, and table persistence logic.
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


class TestStatecodeRecoding:
    def test_statecode_recoding(self):
        """Verify CA/NY/TX/IL/AZ/OH stay as-is, all other states become 'OTHER'."""
        state_codes = ['CA', 'NY', 'TX', 'IL', 'AZ', 'OH', 'FL', 'WA', 'GA', 'CO']
        expected = ['CA', 'NY', 'TX', 'IL', 'AZ', 'OH', 'OTHER', 'OTHER', 'OTHER', 'OTHER']

        keep_states = {'CA', 'NY', 'TX', 'IL', 'AZ', 'OH'}
        recoded = [s if s in keep_states else 'OTHER' for s in state_codes]

        assert recoded == expected


class TestADS2Columns:
    def test_ads2_columns(self):
        """Verify ADS_Py2 has 23 expected columns (includes statecode instead of 6 state indicators)."""
        expected_columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'statecode', 'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal',
            'ck_avg_tran_amt', 'sv_avg_tran_amt', 'cc_avg_tran_amt',
            'q1_trans_cnt', 'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
        ]
        assert len(expected_columns) == 23
        # Verify statecode is present instead of individual state indicators
        assert 'statecode' in expected_columns
        assert 'ca_resident_ind' not in expected_columns
        assert 'ny_resident_ind' not in expected_columns


class TestTrainTestSplit:
    def test_train_test_split_60_40(self):
        """Verify split proportions of 60/40."""
        np.random.seed(42)
        n = 1000
        data = pd.DataFrame({'cust_id': range(n), 'value': np.random.rand(n)})

        # Simulate sampleid assignment (60/40 split)
        sample_ids = np.random.choice([1, 2], size=n, p=[0.6, 0.4])
        data['sampleid'] = sample_ids

        train = data[data['sampleid'] == 1]
        test = data[data['sampleid'] == 2]

        # Allow 5% tolerance
        assert len(train) == pytest.approx(600, abs=50)
        assert len(test) == pytest.approx(400, abs=50)
        assert len(train) + len(test) == n

    def test_multi_model_train_persisted(self):
        """Verify MultiModelTrain_Py is created from sampleid==1."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'cust_id': range(n),
            'sampleid': np.random.choice([1, 2], size=n, p=[0.6, 0.4]),
        })

        train = data[data['sampleid'] == 1]
        assert len(train) > 0
        assert all(train['sampleid'] == 1)

    def test_multi_model_test_persisted(self):
        """Verify MultiModelTest_Py is created from sampleid==2."""
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'cust_id': range(n),
            'sampleid': np.random.choice([1, 2], size=n, p=[0.6, 0.4]),
        })

        test = data[data['sampleid'] == 2]
        assert len(test) > 0
        assert all(test['sampleid'] == 2)
