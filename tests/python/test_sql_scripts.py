"""
Tests for SQL scripts (structural validation).

Since SQL scripts can't be executed without Teradata, these tests
verify structure, syntax elements, and correctness of references.
"""

import os
import pytest

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


class TestPart0SQL:
    @pytest.fixture
    def sql_content(self):
        path = os.path.join(REPO_ROOT, "R_Py-Part_0", "R_Py_TechBytes-Part_0-Demo.sql")
        with open(path, 'r') as f:
            return f.read()

    def test_part0_sql_has_create_table_ads(self, sql_content):
        """Verify SQL contains CREATE TABLE ADS_SQL."""
        assert 'CREATE TABLE ADS_SQL' in sql_content

    def test_part0_sql_has_xgboost(self, sql_content):
        """Verify SQL contains XGBoost call."""
        assert 'XGBoost' in sql_content

    def test_part0_sql_has_decision_forest(self, sql_content):
        """Verify SQL contains DecisionForest call."""
        assert 'DecisionForest' in sql_content

    def test_part0_sql_has_confusion_matrix(self, sql_content):
        """Verify SQL contains ConfusionMatrix call."""
        assert 'ConfusionMatrix' in sql_content

    def test_part0_sql_has_cleanup(self, sql_content):
        """Verify DROP TABLE statements exist for all temp tables."""
        expected_drops = [
            'DROP TABLE tempXGBmodel',
            'DROP TABLE XGBprediction',
            'DROP TABLE tempDFmodel',
            'DROP TABLE tempDFmonitor',
            'DROP TABLE DFevaluation',
            'DROP TABLE DFprediction',
            'DROP TABLE tempXGBcount',
            'DROP TABLE tempXGBstat',
            'DROP TABLE tempXGBaccuracy',
            'DROP TABLE tempRFcount',
            'DROP TABLE tempRFstat',
            'DROP TABLE tempRFaccuracy',
        ]
        for drop in expected_drops:
            assert drop in sql_content, f"Missing cleanup: {drop}"


class TestPart5SQL:
    @pytest.fixture
    def sql_content(self):
        path = os.path.join(REPO_ROOT, "R_Py-Part_5", "R_Py_TechBytes-Part_5-Demo.sql")
        with open(path, 'r') as f:
            return f.read()

    def test_part5_sql_script_command(self, sql_content):
        """Verify SCRIPT_COMMAND references correct Python script paths."""
        assert 'stoRFScore.py' in sql_content
        assert 'stoRFFitMM.py' in sql_content
        assert 'stoRFScoreMM.py' in sql_content

    def test_part5_sql_returns_clause(self, sql_content):
        """Verify RETURNS clause matches expected output schema."""
        # Use Case 1: stoRFScore returns (oc1 INTEGER, oc3 FLOAT, oc4 FLOAT, oc5 INTEGER)
        assert "RETURNS ('oc1 INTEGER, oc3 FLOAT, oc4 FLOAT, oc5 INTEGER')" in sql_content
        # Use Case 2 fit: returns statecode and model
        assert "RETURNS ('oc1 VARCHAR(10), oc2 CLOB')" in sql_content
        # Use Case 2 score: returns cust_id, statecode, Prob0, Prob1, Actual
        assert "RETURNS ('oc1 INTEGER, oc2 VARCHAR(10), oc3 FLOAT, oc4 FLOAT, oc5 INTEGER')" in sql_content

    def test_part5_sql_has_install_file(self, sql_content):
        """Verify SYSUIF.INSTALL_FILE calls for scripts and model."""
        assert 'SYSUIF.INSTALL_FILE' in sql_content
        assert 'SYSUIF.REMOVE_FILE' in sql_content

    def test_part5_sql_partition_by(self, sql_content):
        """Verify multi-model uses PARTITION BY for state-based splitting."""
        assert 'PARTITION BY scode' in sql_content


class TestFastloadScripts:
    def test_fastload_scripts_reference_correct_files(self):
        """Verify each .fastload references the correct .csv file."""
        input_dir = os.path.join(REPO_ROOT, "R_Py-Input_Tables")

        fastload_csv_pairs = [
            ("Accounts.fastload", "Accounts.csv"),
            ("Customer.fastload", "Customer.csv"),
            ("Transactions.fastload", "Transactions.csv"),
        ]

        for fastload_file, csv_file in fastload_csv_pairs:
            path = os.path.join(input_dir, fastload_file)
            with open(path, 'r') as f:
                content = f.read()
            assert csv_file in content, \
                f"{fastload_file} should reference {csv_file}"
