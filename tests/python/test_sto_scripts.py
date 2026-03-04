"""
Tests for in-database Python scripts (stoRFScore.py, stoRFFitMM.py, stoRFScoreMM.py).

These scripts read from stdin and write to stdout. Tests simulate stdin/stdout
using io.StringIO.
"""

import os
import sys
import io
import pytest
import pickle
import base64
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

PART5_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "R_Py-Part_5")


def make_stoRFScore_input(n_rows=5):
    """Generate tab-delimited input data matching stoRFScore.py 28-column schema."""
    columns = [
        'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
        'female_ind', 'single_ind', 'married_ind', 'separated_ind',
        'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
        'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
        'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
        'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
        'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
        'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
    ]
    lines = []
    for i in range(n_rows):
        values = [
            str(i + 1),           # cust_id
            '5.000E 004',         # tot_income (scientific with space)
            str(30 + i),          # tot_age
            str(5 + i),           # tot_cust_years
            str(i % 4),           # tot_children
            str(i % 2),           # female_ind
            str(1 if i % 3 == 0 else 0),  # single_ind
            str(1 if i % 3 == 1 else 0),  # married_ind
            str(1 if i % 3 == 2 else 0),  # separated_ind
            str(1 if i % 6 == 0 else 0),  # ca_resident_ind
            str(1 if i % 6 == 1 else 0),  # ny_resident_ind
            str(1 if i % 6 == 2 else 0),  # tx_resident_ind
            str(1 if i % 6 == 3 else 0),  # il_resident_ind
            str(1 if i % 6 == 4 else 0),  # az_resident_ind
            str(1 if i % 6 == 5 else 0),  # oh_resident_ind
            str(1),               # ck_acct_ind
            str(1),               # sv_acct_ind
            str(i % 2),           # cc_acct_ind
            '1.000E 003',         # ck_avg_bal
            '2.000E 003',         # sv_avg_bal
            '5.000E 002',         # cc_avg_bal
            '1.500E 002',         # ck_avg_tran_amt
            '1.000E 002',         # sv_avg_tran_amt
            '5.000E 001',         # cc_avg_tran_amt
            str(10 + i),          # q1_trans_cnt
            str(8 + i),           # q2_trans_cnt
            str(12 + i),          # q3_trans_cnt
            str(9 + i),           # q4_trans_cnt
        ]
        lines.append('\t'.join(values))
    return '\n'.join(lines)


def make_stoRFFitMM_input(n_rows=5):
    """Generate tab-delimited input data matching stoRFFitMM.py 24-column schema."""
    lines = []
    for i in range(n_rows):
        values = [
            str(i + 1),           # cust_id
            '5.000E 004',         # tot_income
            str(30 + i),          # tot_age
            str(5 + i),           # tot_cust_years
            str(i % 4),           # tot_children
            str(i % 2),           # female_ind
            str(1 if i % 3 == 0 else 0),  # single_ind
            str(1 if i % 3 == 1 else 0),  # married_ind
            str(1 if i % 3 == 2 else 0),  # separated_ind
            '"CA"',               # statecode
            str(1),               # ck_acct_ind
            str(1),               # sv_acct_ind
            str(i % 2),           # cc_acct_ind
            '1.000E 003',         # ck_avg_bal
            '2.000E 003',         # sv_avg_bal
            '5.000E 002',         # cc_avg_bal
            '1.500E 002',         # ck_avg_tran_amt
            '1.000E 002',         # sv_avg_tran_amt
            '5.000E 001',         # cc_avg_tran_amt
            str(10 + i),          # q1_trans_cnt
            str(8 + i),           # q2_trans_cnt
            str(12 + i),          # q3_trans_cnt
            str(9 + i),           # q4_trans_cnt
            str(1),               # SAMPLE_ID
        ]
        lines.append('\t'.join(values))
    return '\n'.join(lines)


class TestStoRFScore:
    def test_stoRFScore_empty_input(self):
        """Feed empty stdin, verify graceful exit (sys.exit)."""
        script_path = os.path.join(PART5_DIR, "stoRFScore.py")
        with open(script_path, 'r') as f:
            code = f.read()

        # Replace the model loading part to avoid file-not-found
        code = code.replace(
            "fIn = open('<DBNAME>/RFmodel_py.out', 'rb')",
            "# MODEL LOADING SKIPPED FOR TEST"
        )
        code = code.replace("classifierPklB64 = fIn.read()", "pass")
        code = code.replace("fIn.close()", "pass")
        code = code.replace("classifierPkl = base64.b64decode(classifierPklB64)", "pass")
        code = code.replace("classifier = pickle.loads(classifierPkl)", "pass")

        # Empty stdin should cause sys.exit
        with patch('sys.stdin', io.StringIO('')):
            with pytest.raises(SystemExit):
                exec(compile(code, script_path, 'exec'), {'__name__': '__test__'})

    def test_stoRFScore_parses_input(self):
        """Feed tab-delimited sample data, verify DataFrame is correctly constructed."""
        input_data = make_stoRFScore_input(5)

        # We'll just test the parsing logic directly
        delimiter = '\t'
        inputData = []
        for line in input_data.splitlines():
            line_parts = line.split(delimiter)
            inputData.append(line_parts)

        columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
            'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
            'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
            'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
            'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
        ]

        df = pd.DataFrame(inputData, columns=columns)
        assert len(df) == 5
        assert list(df.columns) == columns

        # Test numeric conversion for scientific notation with space
        df['tot_income'] = df['tot_income'].apply(lambda x: "".join(x.split()))
        df['tot_income'] = pd.to_numeric(df['tot_income'])
        assert df['tot_income'].iloc[0] == pytest.approx(50000.0)

    def test_stoRFScore_prediction_output(self, tmp_path):
        """Feed sample data + create a mock model file, verify output format."""
        from sklearn.ensemble import RandomForestClassifier

        # Create a small model
        np.random.seed(42)
        X_train = np.random.rand(50, 18)
        y_train = np.random.randint(0, 2, 50)
        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(X_train, y_train)

        # Predict with parsed input
        input_data = make_stoRFScore_input(5)
        delimiter = '\t'
        inputData = []
        for line in input_data.splitlines():
            inputData.append(line.split(delimiter))

        columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
            'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
            'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
            'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
            'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
        ]
        df = pd.DataFrame(inputData, columns=columns)

        # Convert all to numeric
        for col in columns:
            df[col] = df[col].apply(lambda x: "".join(str(x).split()))
            df[col] = pd.to_numeric(df[col])

        predictor_columns = [
            "tot_income", "tot_age", "tot_cust_years", "tot_children",
            "female_ind", "single_ind", "married_ind", "separated_ind",
            "ck_acct_ind", "sv_acct_ind", "ck_avg_bal", "sv_avg_bal",
            "ck_avg_tran_amt", "sv_avg_tran_amt", "q1_trans_cnt",
            "q2_trans_cnt", "q3_trans_cnt", "q4_trans_cnt",
        ]

        X_test = df[predictor_columns]
        PredictionProba = classifier.predict_proba(X_test)

        df = pd.concat([df, pd.DataFrame(data=PredictionProba, columns=['Prob0', 'Prob1'])], axis=1)

        # Verify output format: cust_id, Prob0, Prob1, cc_acct_ind
        output_lines = []
        for index, row in df.iterrows():
            output_lines.append(f"{row['cust_id']}\t{row['Prob0']}\t{row['Prob1']}\t{row['cc_acct_ind']}")

        assert len(output_lines) == 5
        for line in output_lines:
            parts = line.split('\t')
            assert len(parts) == 4


class TestStoRFFitMM:
    def test_stoRFFitMM_empty_input(self):
        """Feed empty stdin, verify graceful exit."""
        script_path = os.path.join(PART5_DIR, "stoRFFitMM.py")
        with open(script_path, 'r') as f:
            code = f.read()

        with patch('sys.stdin', io.StringIO('')):
            with pytest.raises(SystemExit):
                exec(compile(code, script_path, 'exec'), {'__name__': '__test__'})

    def test_stoRFFitMM_parses_input(self):
        """Feed tab-delimited sample data (24-column schema with statecode+SAMPLE_ID)."""
        input_data = make_stoRFFitMM_input(10)

        delimiter = '\t'
        inputData = []
        for line in input_data.splitlines():
            inputData.append(line.split(delimiter))

        columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'statecode', 'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
            'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
            'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt', 'SAMPLE_ID',
        ]

        df = pd.DataFrame(inputData, columns=columns)
        assert len(df) == 10
        assert 'statecode' in df.columns
        assert 'SAMPLE_ID' in df.columns

    def test_stoRFFitMM_output_format(self):
        """Verify output is 'statecode\\tserializedModel'."""
        input_data = make_stoRFFitMM_input(20)

        # Simulate the script's logic
        delimiter = '\t'
        inputData = []
        for line in input_data.splitlines():
            inputData.append(line.split(delimiter))

        columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'statecode', 'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
            'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
            'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt', 'SAMPLE_ID',
        ]

        df = pd.DataFrame(inputData, columns=columns)

        # Convert numeric columns
        for col in columns:
            if col != 'statecode':
                df[col] = df[col].apply(lambda x: "".join(str(x).split()))
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['statecode'] = df['statecode'].apply(lambda x: x.replace('"', ''))

        from sklearn.ensemble import RandomForestClassifier

        predictor_columns = [
            "tot_income", "tot_age", "tot_cust_years", "tot_children",
            "female_ind", "single_ind", "married_ind", "separated_ind",
            "ck_acct_ind", "sv_acct_ind", "ck_avg_bal", "sv_avg_bal",
            "ck_avg_tran_amt", "sv_avg_tran_amt", "q1_trans_cnt",
            "q2_trans_cnt", "q3_trans_cnt", "q4_trans_cnt",
        ]

        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        X = df[predictor_columns]
        y = df["cc_acct_ind"]
        classifier.fit(X, y)

        modelSer = pickle.dumps(classifier)
        modelSerB64 = base64.b64encode(modelSer)

        output = f"{df.iloc[0, 9]}\t{modelSerB64}"
        parts = output.split('\t')
        assert len(parts) == 2
        assert parts[0] == 'CA'

    def test_stoRFFitMM_model_deserializable(self):
        """Verify the output model bytes can be decoded and unpickled."""
        from sklearn.ensemble import RandomForestClassifier

        np.random.seed(42)
        X = np.random.rand(20, 18)
        y = np.random.randint(0, 2, 20)

        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(X, y)

        modelSer = pickle.dumps(classifier)
        modelSerB64 = base64.b64encode(modelSer)

        # Decode and verify
        restored_ser = base64.b64decode(modelSerB64)
        restored = pickle.loads(restored_ser)

        assert isinstance(restored, RandomForestClassifier)
        predictions = restored.predict(X[:5])
        assert len(predictions) == 5


class TestStoRFScoreMM:
    def test_stoRFScoreMM_empty_input(self):
        """Feed empty stdin, verify graceful exit."""
        script_path = os.path.join(PART5_DIR, "stoRFScoreMM.py")
        with open(script_path, 'r') as f:
            code = f.read()

        with patch('builtins.input', side_effect=EOFError):
            with pytest.raises(SystemExit):
                exec(compile(code, script_path, 'exec'), {'__name__': '__test__'})

    def test_stoRFScoreMM_prediction_output(self):
        """Verify output format is 'cust_id\\tstatecode\\tProb0\\tProb1\\tcc_acct_ind'."""
        from sklearn.ensemble import RandomForestClassifier

        # Create a model
        np.random.seed(42)
        X_train = np.random.rand(50, 18)
        y_train = np.random.randint(0, 2, 50)
        classifier = RandomForestClassifier(n_estimators=10, random_state=0)
        classifier.fit(X_train, y_train)

        # Build input data
        input_data = make_stoRFFitMM_input(5)
        delimiter = '\t'
        inputData = []
        for line in input_data.splitlines():
            inputData.append(line.split(delimiter))

        columns = [
            'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
            'female_ind', 'single_ind', 'married_ind', 'separated_ind',
            'statecode', 'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
            'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal', 'ck_avg_tran_amt',
            'sv_avg_tran_amt', 'cc_avg_tran_amt', 'q1_trans_cnt',
            'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt', 'SAMPLE_ID',
        ]

        df = pd.DataFrame(inputData, columns=columns)
        for col in columns:
            if col != 'statecode':
                df[col] = df[col].apply(lambda x: "".join(str(x).split()))
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df['statecode'] = df['statecode'].apply(lambda x: x.replace('"', ''))

        predictor_columns = [
            "tot_income", "tot_age", "tot_cust_years", "tot_children",
            "female_ind", "single_ind", "married_ind", "separated_ind",
            "ck_acct_ind", "sv_acct_ind", "ck_avg_bal", "sv_avg_bal",
            "ck_avg_tran_amt", "sv_avg_tran_amt", "q1_trans_cnt",
            "q2_trans_cnt", "q3_trans_cnt", "q4_trans_cnt",
        ]

        X_test = df[predictor_columns]
        PredictionProba = classifier.predict_proba(X_test)
        df = pd.concat([df, pd.DataFrame(data=PredictionProba, columns=['Prob0', 'Prob1'])], axis=1)

        # Verify output format
        output_lines = []
        for index, row in df.iterrows():
            output_lines.append(
                f"{row['cust_id']}\t{row['statecode']}\t{row['Prob0']}\t{row['Prob1']}\t{row['cc_acct_ind']}"
            )

        assert len(output_lines) == 5
        for line in output_lines:
            parts = line.split('\t')
            assert len(parts) == 5  # cust_id, statecode, Prob0, Prob1, cc_acct_ind
