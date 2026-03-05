#!/usr/bin/env python3
"""
Deployment orchestration script for R and Python TechBytes Demo.

1. Calls the data loading script to populate base tables
2. Runs Part 3 Python demo (feature engineering + ADS creation) to create ADS_Py
3. Runs Part 5 Python demo (Use Case 2 data prep) to create ADS_Py2,
   MultiModelTrain_Py, MultiModelTest_Py
4. Optionally runs model training
5. Reports which tables were created and their row counts
"""

import os
import sys
import argparse

# Allow importing config from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def execute_sql(sql):
    """Execute a raw SQL statement using the current teradataml context."""
    from teradataml import get_context
    from sqlalchemy import text
    with get_context().connect() as conn:
        return conn.execute(text(sql))


def get_table_count(db, table_name):
    """Get row count for a table, return -1 if table doesn't exist."""
    try:
        result = execute_sql(f"SELECT COUNT(*) FROM {db}.{table_name}")
        row = result.fetchone()
        return row[0] if row else -1
    except Exception:
        return -1


def step_load_data():
    """Step 1: Load base data tables."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading base data tables")
    print("=" * 60)
    from scripts.load_data import load_data
    return load_data()


def _build_acct_features_sql(db):
    """SQL fragment for account-level features aggregated by cust_id."""
    return f"""
    SELECT
        a.cust_id,
        MAX(CASE WHEN a.acct_type = 'CK' THEN 1 ELSE 0 END) AS ck_acct_ind,
        MAX(CASE WHEN a.acct_type = 'SV' THEN 1 ELSE 0 END) AS sv_acct_ind,
        MAX(CASE WHEN a.acct_type = 'CC' THEN 1 ELSE 0 END) AS cc_acct_ind,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'CK'
            THEN a.starting_balance + a.ending_balance ELSE 0 END AS FLOAT)) AS FLOAT) AS ck_avg_bal,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'SV'
            THEN a.starting_balance + a.ending_balance ELSE 0 END AS FLOAT)) AS FLOAT) AS sv_avg_bal,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'CC'
            THEN a.starting_balance + a.ending_balance ELSE 0 END AS FLOAT)) AS FLOAT) AS cc_avg_bal
    FROM {db}.Accounts a
    GROUP BY a.cust_id
    """


def _build_tran_features_sql(db):
    """SQL fragment for transaction-level features aggregated by cust_id."""
    return f"""
    SELECT
        a.cust_id,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'CK'
            THEN t.principal_amt + t.interest_amt ELSE 0 END AS FLOAT)) AS FLOAT) AS ck_avg_tran_amt,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'SV'
            THEN t.principal_amt + t.interest_amt ELSE 0 END AS FLOAT)) AS FLOAT) AS sv_avg_tran_amt,
        CAST(AVG(CAST(CASE WHEN a.acct_type = 'CC'
            THEN t.principal_amt + t.interest_amt ELSE 0 END AS FLOAT)) AS FLOAT) AS cc_avg_tran_amt,
        COUNT(CASE WHEN EXTRACT(MONTH FROM t.tran_date) IN (1,2,3)
            THEN t.tran_id ELSE NULL END) AS q1_trans_cnt,
        COUNT(CASE WHEN EXTRACT(MONTH FROM t.tran_date) IN (4,5,6)
            THEN t.tran_id ELSE NULL END) AS q2_trans_cnt,
        COUNT(CASE WHEN EXTRACT(MONTH FROM t.tran_date) IN (7,8,9)
            THEN t.tran_id ELSE NULL END) AS q3_trans_cnt,
        COUNT(CASE WHEN EXTRACT(MONTH FROM t.tran_date) IN (10,11,12)
            THEN t.tran_id ELSE NULL END) AS q4_trans_cnt
    FROM {db}.Accounts a
    LEFT JOIN {db}.Transactions t ON CAST(a.acct_nbr AS VARCHAR(20)) = t.acct_nbr
    GROUP BY a.cust_id
    """


def step_create_ads(db):
    """Step 2: Create ADS_Py via SQL CTAS using staged temp tables.

    Uses pure SQL with CAST to FLOAT to avoid Teradata DECIMAL precision
    overflow errors that occur when using teradataml copy_to_sql with
    complex aggregation expressions.
    """
    print("\n" + "=" * 60)
    print("STEP 2: Creating ADS_Py (Part 3 feature engineering)")
    print("=" * 60)

    # Build account and transaction feature temp tables, then join with
    # customer features to produce the final ADS_Py table.
    for tmp in ["tmp_acct_features", "tmp_tran_features", "tmp_cust_features", "ADS_Py"]:
        try:
            execute_sql(f"DROP TABLE {db}.{tmp}")
        except Exception:
            pass

    print("  Building account features...")
    execute_sql(f"CREATE TABLE {db}.tmp_acct_features AS ({_build_acct_features_sql(db)}) WITH DATA")

    print("  Building transaction features...")
    execute_sql(f"CREATE TABLE {db}.tmp_tran_features AS ({_build_tran_features_sql(db)}) WITH DATA")

    print("  Building customer features...")
    execute_sql(f"""
    CREATE TABLE {db}.tmp_cust_features AS (
        SELECT
            c.cust_id,
            CAST(c.income AS FLOAT) AS tot_income,
            c.age AS tot_age,
            c.years_with_bank AS tot_cust_years,
            c.nbr_children AS tot_children,
            CASE WHEN c.gender = 'F' THEN 1 ELSE 0 END AS female_ind,
            CASE WHEN c.marital_status = 1 THEN 1 ELSE 0 END AS single_ind,
            CASE WHEN c.marital_status = 2 THEN 1 ELSE 0 END AS married_ind,
            CASE WHEN c.marital_status = 3 THEN 1 ELSE 0 END AS separated_ind,
            CASE WHEN c.state_code = 'CA' THEN 1 ELSE 0 END AS ca_resident_ind,
            CASE WHEN c.state_code = 'NY' THEN 1 ELSE 0 END AS ny_resident_ind,
            CASE WHEN c.state_code = 'TX' THEN 1 ELSE 0 END AS tx_resident_ind,
            CASE WHEN c.state_code = 'IL' THEN 1 ELSE 0 END AS il_resident_ind,
            CASE WHEN c.state_code = 'AZ' THEN 1 ELSE 0 END AS az_resident_ind,
            CASE WHEN c.state_code = 'OH' THEN 1 ELSE 0 END AS oh_resident_ind,
            CASE WHEN c.state_code IN ('CA','NY','TX','IL','AZ','OH')
                THEN TRIM(c.state_code) ELSE 'OTHER' END AS statecode
        FROM {db}.Customer c
    ) WITH DATA
    """)

    print("  Joining into ADS_Py...")
    execute_sql(f"""
    CREATE TABLE {db}.ADS_Py AS (
        SELECT
            cf.cust_id,
            cf.tot_income, cf.tot_age, cf.tot_cust_years, cf.tot_children,
            cf.female_ind, cf.single_ind, cf.married_ind, cf.separated_ind,
            cf.ca_resident_ind, cf.ny_resident_ind, cf.tx_resident_ind,
            cf.il_resident_ind, cf.az_resident_ind, cf.oh_resident_ind,
            COALESCE(af.ck_acct_ind, 0) AS ck_acct_ind,
            COALESCE(af.sv_acct_ind, 0) AS sv_acct_ind,
            COALESCE(af.cc_acct_ind, 0) AS cc_acct_ind,
            COALESCE(af.ck_avg_bal, 0.0) AS ck_avg_bal,
            COALESCE(af.sv_avg_bal, 0.0) AS sv_avg_bal,
            COALESCE(af.cc_avg_bal, 0.0) AS cc_avg_bal,
            COALESCE(tf.ck_avg_tran_amt, 0.0) AS ck_avg_tran_amt,
            COALESCE(tf.sv_avg_tran_amt, 0.0) AS sv_avg_tran_amt,
            COALESCE(tf.cc_avg_tran_amt, 0.0) AS cc_avg_tran_amt,
            COALESCE(tf.q1_trans_cnt, 0) AS q1_trans_cnt,
            COALESCE(tf.q2_trans_cnt, 0) AS q2_trans_cnt,
            COALESCE(tf.q3_trans_cnt, 0) AS q3_trans_cnt,
            COALESCE(tf.q4_trans_cnt, 0) AS q4_trans_cnt
        FROM {db}.tmp_cust_features cf
        LEFT JOIN {db}.tmp_acct_features af ON cf.cust_id = af.cust_id
        LEFT JOIN {db}.tmp_tran_features tf ON cf.cust_id = tf.cust_id
    ) WITH DATA
    """)

    count = get_table_count(db, "ADS_Py")
    print(f"  ADS_Py created with {count} rows.")

    # Clean up temp tables
    for tmp in ["tmp_acct_features", "tmp_tran_features"]:
        try:
            execute_sql(f"DROP TABLE {db}.{tmp}")
        except Exception:
            pass

    return count > 0


def step_create_multimodel_tables(db):
    """Step 3: Create ADS_Py2, MultiModelTrain_Py, MultiModelTest_Py.

    Uses the same staged-SQL approach as step_create_ads. ADS_Py2 uses
    statecode (recoded state) instead of individual state indicator columns.
    Train/test split uses deterministic cust_id MOD for ~60/40 partition.
    """
    print("\n" + "=" * 60)
    print("STEP 3: Creating multi-model tables (Part 5 Use Case 2)")
    print("=" * 60)

    for tbl in ["ADS_Py2", "MultiModelTrain_Py", "MultiModelTest_Py"]:
        try:
            execute_sql(f"DROP TABLE {db}.{tbl}")
        except Exception:
            pass

    # Re-use the tmp_cust_features table from step_create_ads if it still
    # exists; otherwise the table was already created above.
    cust_exists = get_table_count(db, "tmp_cust_features")
    if cust_exists < 0:
        print("  Rebuilding customer features...")
        execute_sql(f"""
        CREATE TABLE {db}.tmp_cust_features AS (
            SELECT
                c.cust_id,
                CAST(c.income AS FLOAT) AS tot_income,
                c.age AS tot_age,
                c.years_with_bank AS tot_cust_years,
                c.nbr_children AS tot_children,
                CASE WHEN c.gender = 'F' THEN 1 ELSE 0 END AS female_ind,
                CASE WHEN c.marital_status = 1 THEN 1 ELSE 0 END AS single_ind,
                CASE WHEN c.marital_status = 2 THEN 1 ELSE 0 END AS married_ind,
                CASE WHEN c.marital_status = 3 THEN 1 ELSE 0 END AS separated_ind,
                CASE WHEN c.state_code = 'CA' THEN 1 ELSE 0 END AS ca_resident_ind,
                CASE WHEN c.state_code = 'NY' THEN 1 ELSE 0 END AS ny_resident_ind,
                CASE WHEN c.state_code = 'TX' THEN 1 ELSE 0 END AS tx_resident_ind,
                CASE WHEN c.state_code = 'IL' THEN 1 ELSE 0 END AS il_resident_ind,
                CASE WHEN c.state_code = 'AZ' THEN 1 ELSE 0 END AS az_resident_ind,
                CASE WHEN c.state_code = 'OH' THEN 1 ELSE 0 END AS oh_resident_ind,
                CASE WHEN c.state_code IN ('CA','NY','TX','IL','AZ','OH')
                    THEN TRIM(c.state_code) ELSE 'OTHER' END AS statecode
            FROM {db}.Customer c
        ) WITH DATA
        """)

    acct_exists = get_table_count(db, "tmp_acct_features")
    if acct_exists < 0:
        print("  Rebuilding account features...")
        execute_sql(f"CREATE TABLE {db}.tmp_acct_features AS ({_build_acct_features_sql(db)}) WITH DATA")

    tran_exists = get_table_count(db, "tmp_tran_features")
    if tran_exists < 0:
        print("  Rebuilding transaction features...")
        execute_sql(f"CREATE TABLE {db}.tmp_tran_features AS ({_build_tran_features_sql(db)}) WITH DATA")

    print("  Creating ADS_Py2 (with statecode)...")
    execute_sql(f"""
    CREATE TABLE {db}.ADS_Py2 AS (
        SELECT
            cf.cust_id,
            cf.tot_income, cf.tot_age, cf.tot_cust_years, cf.tot_children,
            cf.female_ind, cf.single_ind, cf.married_ind, cf.separated_ind,
            cf.statecode,
            COALESCE(af.ck_acct_ind, 0) AS ck_acct_ind,
            COALESCE(af.sv_acct_ind, 0) AS sv_acct_ind,
            COALESCE(af.cc_acct_ind, 0) AS cc_acct_ind,
            COALESCE(af.ck_avg_bal, 0.0) AS ck_avg_bal,
            COALESCE(af.sv_avg_bal, 0.0) AS sv_avg_bal,
            COALESCE(af.cc_avg_bal, 0.0) AS cc_avg_bal,
            COALESCE(tf.ck_avg_tran_amt, 0.0) AS ck_avg_tran_amt,
            COALESCE(tf.sv_avg_tran_amt, 0.0) AS sv_avg_tran_amt,
            COALESCE(tf.cc_avg_tran_amt, 0.0) AS cc_avg_tran_amt,
            COALESCE(tf.q1_trans_cnt, 0) AS q1_trans_cnt,
            COALESCE(tf.q2_trans_cnt, 0) AS q2_trans_cnt,
            COALESCE(tf.q3_trans_cnt, 0) AS q3_trans_cnt,
            COALESCE(tf.q4_trans_cnt, 0) AS q4_trans_cnt
        FROM {db}.tmp_cust_features cf
        LEFT JOIN {db}.tmp_acct_features af ON cf.cust_id = af.cust_id
        LEFT JOIN {db}.tmp_tran_features tf ON cf.cust_id = tf.cust_id
    ) WITH DATA
    """)

    # Get ADS_Py2 column list for the train/test CTAS
    from teradataml import get_context
    from sqlalchemy import text
    with get_context().connect() as conn:
        result = conn.execute(text(
            f"SELECT ColumnName FROM DBC.ColumnsV "
            f"WHERE DatabaseName = '{db}' AND TableName = 'ADS_Py2' "
            f"ORDER BY ColumnId"
        ))
        ads2_cols = ", ".join(row[0].strip() for row in result.fetchall())

    # Train set (~60%) — deterministic split via cust_id MOD
    print("  Creating MultiModelTrain_Py (~60%)...")
    execute_sql(f"""
    CREATE TABLE {db}.MultiModelTrain_Py AS (
        SELECT {ads2_cols}, 1 AS sample_id
        FROM {db}.ADS_Py2
        WHERE MOD(cust_id, 10) < 6
    ) WITH DATA
    """)

    # Test set (~40%)
    print("  Creating MultiModelTest_Py (~40%)...")
    execute_sql(f"""
    CREATE TABLE {db}.MultiModelTest_Py AS (
        SELECT {ads2_cols}, 2 AS sample_id
        FROM {db}.ADS_Py2
        WHERE MOD(cust_id, 10) >= 6
    ) WITH DATA
    """)

    # Clean up all temp tables
    for tmp in ["tmp_acct_features", "tmp_tran_features", "tmp_cust_features"]:
        try:
            execute_sql(f"DROP TABLE {db}.{tmp}")
        except Exception:
            pass

    for tbl_name in ["ADS_Py2", "MultiModelTrain_Py", "MultiModelTest_Py"]:
        count = get_table_count(db, tbl_name)
        print(f"  {tbl_name}: {count} rows")

    return True


def report_tables(db):
    """Report all tables and their row counts."""
    print("\n" + "=" * 60)
    print("DEPLOYMENT SUMMARY")
    print("=" * 60)

    expected_tables = [
        "Customer", "Accounts", "Transactions",
        "ADS_Py", "ADS_Py2",
        "MultiModelTrain_Py", "MultiModelTest_Py",
    ]

    for table_name in expected_tables:
        count = get_table_count(db, table_name)
        status = f"{count} rows" if count >= 0 else "NOT FOUND"
        print(f"  {table_name}: {status}")


def main():
    parser = argparse.ArgumentParser(description="Deploy TechBytes demo data and tables")
    parser.add_argument("--skip-load", action="store_true", help="Skip base data loading")
    parser.add_argument("--skip-ads", action="store_true", help="Skip ADS creation")
    parser.add_argument("--skip-multimodel", action="store_true", help="Skip multi-model table creation")
    args = parser.parse_args()

    db = config.TD_DATABASE

    from teradataml import create_context
    if not args.skip_load:
        # step_load_data -> load_data() creates its own context, so skip
        # creating one here to avoid orphaning a connection.
        step_load_data()
    else:
        print(f"Connecting to Teradata at {config.TD_HOST}...")
        create_context(host=config.TD_HOST, username=config.TD_USER, password=config.TD_PASSWORD)
        print("  Connected.")

    if not args.skip_ads:
        step_create_ads(db)

    if not args.skip_multimodel:
        step_create_multimodel_tables(db)
    else:
        # Clean up tmp_cust_features left by step_create_ads when
        # the multimodel step (which normally cleans it) is skipped.
        try:
            execute_sql(f"DROP TABLE {db}.tmp_cust_features")
        except Exception:
            pass

    report_tables(db)

    from teradataml import remove_context
    remove_context()
    print("\nDeployment complete.")


if __name__ == "__main__":
    main()
