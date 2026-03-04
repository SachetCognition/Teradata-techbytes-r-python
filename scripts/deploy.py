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


def get_table_count(db, table_name):
    """Get row count for a table, return -1 if table doesn't exist."""
    from teradataml import DataFrame as TDDataFrame, in_schema, get_context
    try:
        td_df = TDDataFrame(in_schema(db, table_name))
        return len(td_df)
    except Exception:
        return -1


def step_load_data():
    """Step 1: Load base data tables."""
    print("\n" + "=" * 60)
    print("STEP 1: Loading base data tables")
    print("=" * 60)
    from scripts.load_data import load_data
    return load_data()


def step_create_ads(db):
    """Step 2: Run Part 3 feature engineering to create ADS_Py table."""
    print("\n" + "=" * 60)
    print("STEP 2: Creating ADS_Py (Part 3 feature engineering)")
    print("=" * 60)

    from teradataml import (
        create_context, DataFrame as TDDataFrame, get_context,
        copy_to_sql, in_schema
    )
    from teradataml.dataframe.sql_functions import case
    from sqlalchemy.sql.expression import extract

    tdCustomer = TDDataFrame(in_schema(db, "Customer"))
    tdAccounts = TDDataFrame(in_schema(db, "Accounts"))
    tdTransactions = TDDataFrame(in_schema(db, "Transactions"))

    # Customer features
    cust = tdCustomer.assign(
        female=case([(tdCustomer.gender == "F", 1)], else_=0),
        single=case([(tdCustomer.marital_status == 1, 1)], else_=0),
        married=case([(tdCustomer.marital_status == 2, 1)], else_=0),
        separated=case([(tdCustomer.marital_status == 3, 1)], else_=0),
        ca_resident=case([(tdCustomer.state_code == "CA", 1)], else_=0),
        ny_resident=case([(tdCustomer.state_code == "NY", 1)], else_=0),
        tx_resident=case([(tdCustomer.state_code == "TX", 1)], else_=0),
        il_resident=case([(tdCustomer.state_code == "IL", 1)], else_=0),
        az_resident=case([(tdCustomer.state_code == "AZ", 1)], else_=0),
        oh_resident=case([(tdCustomer.state_code == "OH", 1)], else_=0),
    )

    # Account features
    acct_balance = tdAccounts.starting_balance + tdAccounts.ending_balance
    acct = tdAccounts.assign(
        ck_acct=case([(tdAccounts.acct_type == "CK", 1)], else_=0),
        sv_acct=case([(tdAccounts.acct_type == "SV", 1)], else_=0),
        cc_acct=case([(tdAccounts.acct_type == "CC", 1)], else_=0),
        ck_bal=case([(tdAccounts.acct_type == "CK", acct_balance.expression)], else_=0),
        sv_bal=case([(tdAccounts.acct_type == "SV", acct_balance.expression)], else_=0),
        cc_bal=case([(tdAccounts.acct_type == "CC", acct_balance.expression)], else_=0),
    )

    # Transaction features
    acct_mon = extract('month', tdTransactions.tran_date.expression).expression
    trans = tdTransactions.assign(
        q1_trans=case([(acct_mon == "1", 1), (acct_mon == "2", 1), (acct_mon == "3", 1)], else_=0),
        q2_trans=case([(acct_mon == "4", 1), (acct_mon == "5", 1), (acct_mon == "6", 1)], else_=0),
        q3_trans=case([(acct_mon == "7", 1), (acct_mon == "8", 1), (acct_mon == "9", 1)], else_=0),
        q4_trans=case([(acct_mon == "10", 1), (acct_mon == "11", 1), (acct_mon == "12", 1)], else_=0),
    )

    # Join accounts and transactions
    acct_trans_cols = [
        'cust_id', 'acct_type', 'starting_balance', 'ending_balance',
        'acct_acct_nbr', 'principal_amt', 'interest_amt', 'tran_id',
        'tran_date', 'q1_trans', 'q2_trans', 'q3_trans', 'q4_trans',
        'cc_acct', 'cc_bal', 'ck_acct', 'ck_bal', 'sv_acct', 'sv_bal'
    ]
    acct_trans_tmp = acct.join(
        other=trans, on=[acct.acct_nbr == trans.acct_nbr],
        how="left", lsuffix="acct", rsuffix="trans"
    ).select(acct_trans_cols)

    acct_trans_amt = trans.principal_amt + trans.interest_amt
    acct_trans = acct_trans_tmp.assign(
        ck_tran_amt=case([(acct_trans_tmp.acct_type == "CK", acct_trans_amt.expression)], else_=0),
        sv_tran_amt=case([(acct_trans_tmp.acct_type == "SV", acct_trans_amt.expression)], else_=0),
        cc_tran_amt=case([(acct_trans_tmp.acct_type == "CC", acct_trans_amt.expression)], else_=0),
    )

    # Join with customer
    ADS_Py_join_tmp = cust.join(
        other=acct_trans, on=[cust.cust_id == acct_trans.cust_id],
        how="left", lsuffix="cust", rsuffix="actr"
    )

    ADS_Py_join = ADS_Py_join_tmp.assign(
        drop_columns=True,
        cust_id=ADS_Py_join_tmp.cust_cust_id,
        income=ADS_Py_join_tmp.income,
        age=ADS_Py_join_tmp.age,
        years_with_bank=ADS_Py_join_tmp.years_with_bank,
        nbr_children=ADS_Py_join_tmp.nbr_children,
        female=ADS_Py_join_tmp.female,
        single=ADS_Py_join_tmp.single,
        married=ADS_Py_join_tmp.married,
        separated=ADS_Py_join_tmp.separated,
        ca_resident=ADS_Py_join_tmp.ca_resident,
        ny_resident=ADS_Py_join_tmp.ny_resident,
        tx_resident=ADS_Py_join_tmp.tx_resident,
        il_resident=ADS_Py_join_tmp.il_resident,
        az_resident=ADS_Py_join_tmp.az_resident,
        oh_resident=ADS_Py_join_tmp.oh_resident,
        ck_acct=case([(ADS_Py_join_tmp.ck_acct == None, 0)], else_=ADS_Py_join_tmp.ck_acct),
        sv_acct=case([(ADS_Py_join_tmp.sv_acct == None, 0)], else_=ADS_Py_join_tmp.sv_acct),
        cc_acct=case([(ADS_Py_join_tmp.cc_acct == None, 0)], else_=ADS_Py_join_tmp.cc_acct),
        ck_bal=case([(ADS_Py_join_tmp.ck_bal == None, 0)], else_=ADS_Py_join_tmp.ck_bal),
        sv_bal=case([(ADS_Py_join_tmp.sv_bal == None, 0)], else_=ADS_Py_join_tmp.sv_bal),
        cc_bal=case([(ADS_Py_join_tmp.cc_bal == None, 0)], else_=ADS_Py_join_tmp.cc_bal),
        ck_tran_amt=case([(ADS_Py_join_tmp.ck_tran_amt == None, 0)], else_=ADS_Py_join_tmp.ck_tran_amt),
        sv_tran_amt=case([(ADS_Py_join_tmp.sv_tran_amt == None, 0)], else_=ADS_Py_join_tmp.sv_tran_amt),
        cc_tran_amt=case([(ADS_Py_join_tmp.cc_tran_amt == None, 0)], else_=ADS_Py_join_tmp.cc_tran_amt),
        q1_trans=case([(ADS_Py_join_tmp.q1_trans == None, 0)], else_=ADS_Py_join_tmp.q1_trans),
        q2_trans=case([(ADS_Py_join_tmp.q2_trans == None, 0)], else_=ADS_Py_join_tmp.q2_trans),
        q3_trans=case([(ADS_Py_join_tmp.q3_trans == None, 0)], else_=ADS_Py_join_tmp.q3_trans),
        q4_trans=case([(ADS_Py_join_tmp.q4_trans == None, 0)], else_=ADS_Py_join_tmp.q4_trans),
    )

    # Aggregate
    ADS_Py = ADS_Py_join.groupby("cust_id").agg({
        "income": "min", "age": "min", "years_with_bank": "min",
        "nbr_children": "min", "single": "min", "female": "min",
        "married": "min", "separated": "min",
        "ca_resident": "max", "ny_resident": "max", "tx_resident": "max",
        "il_resident": "max", "az_resident": "max", "oh_resident": "max",
        "ck_acct": "max", "sv_acct": "max", "cc_acct": "max",
        "ck_bal": "mean", "sv_bal": "mean", "cc_bal": "mean",
        "ck_tran_amt": "mean", "sv_tran_amt": "mean", "cc_tran_amt": "mean",
        "q1_trans": "count", "q2_trans": "count", "q3_trans": "count", "q4_trans": "count",
    })

    columns = [
        'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
        'female_ind', 'single_ind', 'married_ind', 'separated_ind',
        'ca_resident_ind', 'ny_resident_ind', 'tx_resident_ind',
        'il_resident_ind', 'az_resident_ind', 'oh_resident_ind',
        'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
        'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal',
        'ck_avg_tran_amt', 'sv_avg_tran_amt', 'cc_avg_tran_amt',
        'q1_trans_cnt', 'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
    ]

    ADS_Py = ADS_Py.assign(
        drop_columns=True,
        cust_id=ADS_Py.cust_id,
        tot_income=ADS_Py.min_income, tot_age=ADS_Py.min_age,
        tot_cust_years=ADS_Py.min_years_with_bank,
        tot_children=ADS_Py.min_nbr_children,
        female_ind=ADS_Py.min_female, single_ind=ADS_Py.min_single,
        married_ind=ADS_Py.min_married, separated_ind=ADS_Py.min_separated,
        ca_resident_ind=ADS_Py.max_ca_resident,
        ny_resident_ind=ADS_Py.max_ny_resident,
        tx_resident_ind=ADS_Py.max_tx_resident,
        il_resident_ind=ADS_Py.max_il_resident,
        az_resident_ind=ADS_Py.max_az_resident,
        oh_resident_ind=ADS_Py.max_oh_resident,
        ck_acct_ind=ADS_Py.max_ck_acct, sv_acct_ind=ADS_Py.max_sv_acct,
        cc_acct_ind=ADS_Py.max_cc_acct,
        ck_avg_bal=ADS_Py.mean_ck_bal, sv_avg_bal=ADS_Py.mean_sv_bal,
        cc_avg_bal=ADS_Py.mean_cc_bal,
        ck_avg_tran_amt=ADS_Py.mean_ck_tran_amt,
        sv_avg_tran_amt=ADS_Py.mean_sv_tran_amt,
        cc_avg_tran_amt=ADS_Py.mean_cc_tran_amt,
        q1_trans_cnt=ADS_Py.count_q1_trans, q2_trans_cnt=ADS_Py.count_q2_trans,
        q3_trans_cnt=ADS_Py.count_q3_trans, q4_trans_cnt=ADS_Py.count_q4_trans,
    ).select(columns)

    # Persist
    try:
        get_context().execute(f"DROP TABLE {db}.ADS_Py")
    except Exception:
        pass
    copy_to_sql(ADS_Py, schema_name=db, table_name="ADS_Py", if_exists="replace")
    count = get_table_count(db, "ADS_Py")
    print(f"  ADS_Py created with {count} rows.")
    return count > 0


def step_create_multimodel_tables(db):
    """Step 3: Create ADS_Py2, MultiModelTrain_Py, MultiModelTest_Py."""
    print("\n" + "=" * 60)
    print("STEP 3: Creating multi-model tables (Part 5 Use Case 2)")
    print("=" * 60)

    from teradataml import (
        DataFrame as TDDataFrame, get_context, copy_to_sql, in_schema
    )
    from teradataml.dataframe.sql_functions import case
    from sqlalchemy.sql.expression import extract

    tdCustomer = TDDataFrame(in_schema(db, "Customer"))
    tdAccounts = TDDataFrame(in_schema(db, "Accounts"))
    tdTransactions = TDDataFrame(in_schema(db, "Transactions"))

    # Customer features with statecode
    cust = tdCustomer.assign(
        drop_columns=True,
        cust_id=tdCustomer.cust_id, income=tdCustomer.income,
        age=tdCustomer.age, gender=tdCustomer.gender,
        years_with_bank=tdCustomer.years_with_bank,
        nbr_children=tdCustomer.nbr_children,
        marital_status=tdCustomer.marital_status,
        state_code=tdCustomer.state_code,
        female=case([(tdCustomer.gender == "F", 1)], else_=0),
        single=case([(tdCustomer.marital_status == 1, 1)], else_=0),
        married=case([(tdCustomer.marital_status == 2, 1)], else_=0),
        separated=case([(tdCustomer.marital_status == 3, 1)], else_=0),
        statecode=case([
            (tdCustomer.state_code == "CA", "CA"),
            (tdCustomer.state_code == "NY", "NY"),
            (tdCustomer.state_code == "TX", "TX"),
            (tdCustomer.state_code == "IL", "IL"),
            (tdCustomer.state_code == "AZ", "AZ"),
            (tdCustomer.state_code == "OH", "OH"),
        ], else_="OTHER"),
    )

    acct_balance = tdAccounts.starting_balance + tdAccounts.ending_balance
    acct = tdAccounts.assign(
        ck_acct=case([(tdAccounts.acct_type == "CK", 1)], else_=0),
        sv_acct=case([(tdAccounts.acct_type == "SV", 1)], else_=0),
        cc_acct=case([(tdAccounts.acct_type == "CC", 1)], else_=0),
        ck_bal=case([(tdAccounts.acct_type == "CK", acct_balance.expression)], else_=0),
        sv_bal=case([(tdAccounts.acct_type == "SV", acct_balance.expression)], else_=0),
        cc_bal=case([(tdAccounts.acct_type == "CC", acct_balance.expression)], else_=0),
    )

    acct_mon = extract('month', tdTransactions.tran_date.expression).expression
    trans = tdTransactions.assign(
        q1_trans=case([(acct_mon == "1", 1), (acct_mon == "2", 1), (acct_mon == "3", 1)], else_=0),
        q2_trans=case([(acct_mon == "4", 1), (acct_mon == "5", 1), (acct_mon == "6", 1)], else_=0),
        q3_trans=case([(acct_mon == "7", 1), (acct_mon == "8", 1), (acct_mon == "9", 1)], else_=0),
        q4_trans=case([(acct_mon == "10", 1), (acct_mon == "11", 1), (acct_mon == "12", 1)], else_=0),
    )

    acct_trans_cols = [
        'cust_id', 'acct_type', 'starting_balance', 'ending_balance',
        'acct_acct_nbr', 'principal_amt', 'interest_amt', 'tran_id',
        'tran_date', 'q1_trans', 'q2_trans', 'q3_trans', 'q4_trans',
        'cc_acct', 'cc_bal', 'ck_acct', 'ck_bal', 'sv_acct', 'sv_bal'
    ]
    acct_trans_tmp = acct.join(
        other=trans, on=[acct.acct_nbr == trans.acct_nbr],
        how="left", lsuffix="acct", rsuffix="trans"
    ).select(acct_trans_cols)

    acct_trans_amt = trans.principal_amt + trans.interest_amt
    acct_trans = acct_trans_tmp.assign(
        ck_tran_amt=case([(acct_trans_tmp.acct_type == "CK", acct_trans_amt.expression)], else_=0),
        sv_tran_amt=case([(acct_trans_tmp.acct_type == "SV", acct_trans_amt.expression)], else_=0),
        cc_tran_amt=case([(acct_trans_tmp.acct_type == "CC", acct_trans_amt.expression)], else_=0),
    )

    ADS_Py2_join_tmp = cust.join(
        other=acct_trans, on=[cust.cust_id == acct_trans.cust_id],
        how="left", lsuffix="cust", rsuffix="actr"
    )

    ADS_Py2_join = ADS_Py2_join_tmp.assign(
        drop_columns=True,
        cust_id=ADS_Py2_join_tmp.cust_cust_id,
        income=ADS_Py2_join_tmp.income, age=ADS_Py2_join_tmp.age,
        years_with_bank=ADS_Py2_join_tmp.years_with_bank,
        nbr_children=ADS_Py2_join_tmp.nbr_children,
        female=ADS_Py2_join_tmp.female,
        single=ADS_Py2_join_tmp.single,
        married=ADS_Py2_join_tmp.married,
        separated=ADS_Py2_join_tmp.separated,
        statecode=ADS_Py2_join_tmp.statecode,
        ck_acct=case([(ADS_Py2_join_tmp.ck_acct == None, 0)], else_=ADS_Py2_join_tmp.ck_acct),
        sv_acct=case([(ADS_Py2_join_tmp.sv_acct == None, 0)], else_=ADS_Py2_join_tmp.sv_acct),
        cc_acct=case([(ADS_Py2_join_tmp.cc_acct == None, 0)], else_=ADS_Py2_join_tmp.cc_acct),
        ck_bal=case([(ADS_Py2_join_tmp.ck_bal == None, 0)], else_=ADS_Py2_join_tmp.ck_bal),
        sv_bal=case([(ADS_Py2_join_tmp.sv_bal == None, 0)], else_=ADS_Py2_join_tmp.sv_bal),
        cc_bal=case([(ADS_Py2_join_tmp.cc_bal == None, 0)], else_=ADS_Py2_join_tmp.cc_bal),
        ck_tran_amt=case([(ADS_Py2_join_tmp.ck_tran_amt == None, 0)], else_=ADS_Py2_join_tmp.ck_tran_amt),
        sv_tran_amt=case([(ADS_Py2_join_tmp.sv_tran_amt == None, 0)], else_=ADS_Py2_join_tmp.sv_tran_amt),
        cc_tran_amt=case([(ADS_Py2_join_tmp.cc_tran_amt == None, 0)], else_=ADS_Py2_join_tmp.cc_tran_amt),
        q1_trans=case([(ADS_Py2_join_tmp.q1_trans == None, 0)], else_=ADS_Py2_join_tmp.q1_trans),
        q2_trans=case([(ADS_Py2_join_tmp.q2_trans == None, 0)], else_=ADS_Py2_join_tmp.q2_trans),
        q3_trans=case([(ADS_Py2_join_tmp.q3_trans == None, 0)], else_=ADS_Py2_join_tmp.q3_trans),
        q4_trans=case([(ADS_Py2_join_tmp.q4_trans == None, 0)], else_=ADS_Py2_join_tmp.q4_trans),
    )

    ADS_Py2 = ADS_Py2_join.groupby("cust_id").agg({
        "income": "min", "age": "min", "years_with_bank": "min",
        "nbr_children": "min", "single": "min", "female": "min",
        "married": "min", "separated": "min", "statecode": "min",
        "ck_acct": "max", "sv_acct": "max", "cc_acct": "max",
        "ck_bal": "mean", "sv_bal": "mean", "cc_bal": "mean",
        "ck_tran_amt": "mean", "sv_tran_amt": "mean", "cc_tran_amt": "mean",
        "q1_trans": "count", "q2_trans": "count", "q3_trans": "count", "q4_trans": "count",
    })

    columns = [
        'cust_id', 'tot_income', 'tot_age', 'tot_cust_years', 'tot_children',
        'female_ind', 'single_ind', 'married_ind', 'separated_ind',
        'statecode', 'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
        'ck_avg_bal', 'sv_avg_bal', 'cc_avg_bal',
        'ck_avg_tran_amt', 'sv_avg_tran_amt', 'cc_avg_tran_amt',
        'q1_trans_cnt', 'q2_trans_cnt', 'q3_trans_cnt', 'q4_trans_cnt',
    ]

    ADS_Py2 = ADS_Py2.assign(
        drop_columns=True,
        cust_id=ADS_Py2.cust_id,
        tot_income=ADS_Py2.min_income, tot_age=ADS_Py2.min_age,
        tot_cust_years=ADS_Py2.min_years_with_bank,
        tot_children=ADS_Py2.min_nbr_children,
        female_ind=ADS_Py2.min_female, single_ind=ADS_Py2.min_single,
        married_ind=ADS_Py2.min_married, separated_ind=ADS_Py2.min_separated,
        statecode=ADS_Py2.min_statecode,
        ck_acct_ind=ADS_Py2.max_ck_acct, sv_acct_ind=ADS_Py2.max_sv_acct,
        cc_acct_ind=ADS_Py2.max_cc_acct,
        ck_avg_bal=ADS_Py2.mean_ck_bal, sv_avg_bal=ADS_Py2.mean_sv_bal,
        cc_avg_bal=ADS_Py2.mean_cc_bal,
        ck_avg_tran_amt=ADS_Py2.mean_ck_tran_amt,
        sv_avg_tran_amt=ADS_Py2.mean_sv_tran_amt,
        cc_avg_tran_amt=ADS_Py2.mean_cc_tran_amt,
        q1_trans_cnt=ADS_Py2.count_q1_trans, q2_trans_cnt=ADS_Py2.count_q2_trans,
        q3_trans_cnt=ADS_Py2.count_q3_trans, q4_trans_cnt=ADS_Py2.count_q4_trans,
    ).select(columns)

    # Persist ADS_Py2
    try:
        get_context().execute(f"DROP TABLE {db}.ADS_Py2")
    except Exception:
        pass
    copy_to_sql(ADS_Py2, schema_name=db, table_name="ADS_Py2", if_exists="replace")

    # Create train/test split
    tdADS_Py2 = TDDataFrame(in_schema(db, "ADS_Py2"))
    ADS_Train_Test2 = tdADS_Py2.sample(frac=[0.60, 0.40])
    try:
        get_context().execute(f"DROP TABLE {db}.ADS_Train_Test2")
    except Exception:
        pass
    copy_to_sql(ADS_Train_Test2, schema_name=db, table_name="ADS_Train_Test2", if_exists="replace")

    tdTrain_Test2 = TDDataFrame(in_schema(db, "ADS_Train_Test2"))

    # Train set
    MultiModelTrain_Py = tdTrain_Test2[tdTrain_Test2.sampleid == "1"]
    try:
        get_context().execute(f"DROP TABLE {db}.MultiModelTrain_Py")
    except Exception:
        pass
    copy_to_sql(MultiModelTrain_Py, schema_name=db, table_name="MultiModelTrain_Py", if_exists="replace")

    # Test set
    MultiModelTest_Py = tdTrain_Test2[tdTrain_Test2.sampleid == "2"]
    try:
        get_context().execute(f"DROP TABLE {db}.MultiModelTest_Py")
    except Exception:
        pass
    copy_to_sql(MultiModelTest_Py, schema_name=db, table_name="MultiModelTest_Py", if_exists="replace")

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
    print(f"Connecting to Teradata at {config.TD_HOST}...")
    create_context(host=config.TD_HOST, username=config.TD_USER, password=config.TD_PASSWORD)
    print("  Connected.")

    if not args.skip_load:
        step_load_data()

    if not args.skip_ads:
        step_create_ads(db)

    if not args.skip_multimodel:
        step_create_multimodel_tables(db)

    report_tables(db)

    from teradataml import remove_context
    remove_context()
    print("\nDeployment complete.")


if __name__ == "__main__":
    main()
