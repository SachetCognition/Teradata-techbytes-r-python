#!/usr/bin/env python3
"""
Automated data loading script for R and Python TechBytes Demo.

Reads environment variables, connects to Teradata, creates tables,
loads data from CSV files, and verifies row counts.
"""

import os
import sys
import zipfile

import pandas as pd

# Try to load from .env file if python-dotenv is available
# This MUST happen before importing config, since config.py reads os.environ at import time.
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

# Allow importing config from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


INPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "R_Py-Input_Tables")

# DDL definitions for the 3 tables
CUSTOMER_DDL = """
CREATE SET TABLE {db}.Customer ,FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO
     (
      cust_id INTEGER,
      income DECIMAL(15,1),
      age INTEGER,
      years_with_bank INTEGER,
      nbr_children INTEGER,
      gender CHAR(1) CHARACTER SET LATIN NOT CASESPECIFIC,
      marital_status CHAR(1) CHARACTER SET LATIN NOT CASESPECIFIC,
      postal_code CHAR(5) CHARACTER SET LATIN NOT CASESPECIFIC,
      state_code CHAR(2) CHARACTER SET LATIN NOT CASESPECIFIC)
UNIQUE PRIMARY INDEX ( cust_id )
"""

ACCOUNTS_DDL = """
CREATE SET TABLE {db}.Accounts ,FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO
     (
      acct_nbr VARCHAR(18) CHARACTER SET LATIN NOT CASESPECIFIC,
      cust_id INTEGER,
      acct_type CHAR(2) CHARACTER SET LATIN NOT CASESPECIFIC,
      account_active CHAR(1) CHARACTER SET LATIN NOT CASESPECIFIC,
      acct_start_date DATE FORMAT 'YY/MM/DD',
      starting_balance DECIMAL(11,3),
      ending_balance DECIMAL(11,3))
UNIQUE PRIMARY INDEX ( acct_nbr )
"""

TRANSACTIONS_DDL = """
CREATE SET TABLE {db}.Transactions ,FALLBACK ,
     NO BEFORE JOURNAL,
     NO AFTER JOURNAL,
     CHECKSUM = DEFAULT,
     DEFAULT MERGEBLOCKRATIO
     (
      tran_id INTEGER,
      acct_nbr VARCHAR(18) CHARACTER SET LATIN NOT CASESPECIFIC,
      tran_amt DECIMAL(9,2),
      principal_amt DECIMAL(15,2),
      interest_amt DECIMAL(11,3),
      new_balance DECIMAL(9,2),
      tran_date DATE FORMAT 'YY/MM/DD',
      tran_time INTEGER,
      channel CHAR(1) CHARACTER SET LATIN NOT CASESPECIFIC,
      tran_code CHAR(2) CHARACTER SET LATIN NOT CASESPECIFIC)
UNIQUE PRIMARY INDEX ( tran_id ,acct_nbr )
"""


def unzip_transactions():
    """Unzip Transactions.csv.zip if Transactions.csv doesn't exist."""
    csv_path = os.path.join(INPUT_DIR, "Transactions.csv")
    zip_path = os.path.join(INPUT_DIR, "Transactions.csv.zip")
    if not os.path.exists(csv_path):
        print("Unzipping Transactions.csv.zip...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(INPUT_DIR)
        print("  Done.")
    else:
        print("Transactions.csv already exists, skipping unzip.")


def load_data():
    """Connect to Teradata and load all 3 tables from CSV files."""
    from teradataml import create_context, copy_to_sql, get_context

    db = config.TD_DATABASE

    print(f"Connecting to Teradata at {config.TD_HOST}...")
    td_context = create_context(
        host=config.TD_HOST,
        username=config.TD_USER,
        password=config.TD_PASSWORD
    )
    print("  Connected.")

    # Unzip transactions if needed
    unzip_transactions()

    # Table definitions: (name, ddl, csv_file, expected_approx_rows)
    tables = [
        ("Customer", CUSTOMER_DDL, "Customer.csv", 10000),
        ("Accounts", ACCOUNTS_DDL, "Accounts.csv", 100000),
        ("Transactions", TRANSACTIONS_DDL, "Transactions.csv", 1000000),
    ]

    results = []
    for table_name, ddl, csv_file, expected_rows in tables:
        print(f"\nProcessing {table_name}...")

        # Drop existing table
        try:
            get_context().execute(f"DROP TABLE {db}.{table_name}")
            print(f"  Dropped existing {table_name} table.")
        except Exception:
            print(f"  No existing {table_name} table to drop.")

        # Create table
        try:
            get_context().execute(ddl.format(db=db))
            print(f"  Created {table_name} table.")
        except Exception as e:
            print(f"  Note: Could not create table via DDL ({e}), will use copy_to_sql.")

        # Read CSV
        csv_path = os.path.join(INPUT_DIR, csv_file)
        print(f"  Reading {csv_file}...")
        df = pd.read_csv(csv_path)
        print(f"  Read {len(df)} rows from CSV.")

        # Load into Teradata
        print(f"  Loading into {db}.{table_name}...")
        try:
            copy_to_sql(df, schema_name=db, table_name=table_name, if_exists="replace")
            print(f"  Loaded {len(df)} rows.")
        except Exception as e:
            print(f"  Error loading {table_name}: {e}")
            results.append((table_name, "FAILED", 0, expected_rows))
            continue

        # Verify row count
        try:
            from teradataml import DataFrame as TDDataFrame, in_schema
            td_df = TDDataFrame(in_schema(db, table_name))
            actual_count = len(td_df)
            status = "OK" if actual_count > 0 else "EMPTY"
            results.append((table_name, status, actual_count, expected_rows))
            print(f"  Verified: {actual_count} rows in {table_name}.")
        except Exception as e:
            print(f"  Could not verify row count: {e}")
            results.append((table_name, "UNVERIFIED", len(df), expected_rows))

    # Print summary
    print("\n" + "=" * 60)
    print("DATA LOADING SUMMARY")
    print("=" * 60)
    all_ok = True
    for table_name, status, actual, expected in results:
        indicator = "OK" if status == "OK" else "!!"
        print(f"  [{indicator}] {table_name}: {status} ({actual} rows, expected ~{expected})")
        if status not in ("OK", "UNVERIFIED"):
            all_ok = False

    if all_ok:
        print("\nAll tables loaded successfully!")
    else:
        print("\nSome tables failed to load. Please check the errors above.")

    return all_ok


if __name__ == "__main__":
    success = load_data()
    sys.exit(0 if success else 1)
