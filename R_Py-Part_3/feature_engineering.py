"""
Extracted feature engineering logic from R_Py_TechBytes-Part_3-Demo.py.

These functions operate on pandas DataFrames (no Teradata dependency)
so they can be tested independently.
"""

import pandas as pd
import numpy as np


def create_customer_features(df):
    """
    Create indicator variables from customer demographics.

    Creates: female_ind, single_ind, married_ind, separated_ind,
             ca_resident_ind, ny_resident_ind, tx_resident_ind,
             il_resident_ind, az_resident_ind, oh_resident_ind

    Parameters
    ----------
    df : pd.DataFrame
        Customer DataFrame with columns: gender, marital_status, state_code

    Returns
    -------
    pd.DataFrame
        DataFrame with added indicator columns
    """
    result = df.copy()
    result['female_ind'] = (result['gender'] == 'F').astype(int)
    result['single_ind'] = (result['marital_status'] == 1).astype(int)
    result['married_ind'] = (result['marital_status'] == 2).astype(int)
    result['separated_ind'] = (result['marital_status'] == 3).astype(int)

    for state, col in [('CA', 'ca_resident_ind'), ('NY', 'ny_resident_ind'),
                        ('TX', 'tx_resident_ind'), ('IL', 'il_resident_ind'),
                        ('AZ', 'az_resident_ind'), ('OH', 'oh_resident_ind')]:
        result[col] = (result['state_code'] == state).astype(int)

    return result


def create_account_features(df):
    """
    Create account type indicator and balance features.

    Creates: ck_acct_ind, sv_acct_ind, cc_acct_ind,
             ck_bal, sv_bal, cc_bal

    Parameters
    ----------
    df : pd.DataFrame
        Accounts DataFrame with columns: acct_type, starting_balance, ending_balance

    Returns
    -------
    pd.DataFrame
        DataFrame with added account feature columns
    """
    result = df.copy()
    balance = result['starting_balance'] + result['ending_balance']

    result['ck_acct_ind'] = (result['acct_type'] == 'CK').astype(int)
    result['sv_acct_ind'] = (result['acct_type'] == 'SV').astype(int)
    result['cc_acct_ind'] = (result['acct_type'] == 'CC').astype(int)

    result['ck_bal'] = np.where(result['acct_type'] == 'CK', balance, 0)
    result['sv_bal'] = np.where(result['acct_type'] == 'SV', balance, 0)
    result['cc_bal'] = np.where(result['acct_type'] == 'CC', balance, 0)

    return result


def create_transaction_features(df):
    """
    Create quarterly transaction indicator features.

    Creates: q1_trans, q2_trans, q3_trans, q4_trans,
             ck_tran_amt, sv_tran_amt, cc_tran_amt

    Parameters
    ----------
    df : pd.DataFrame
        Transactions DataFrame joined with Accounts, containing:
        tran_date, acct_type, principal_amt, interest_amt

    Returns
    -------
    pd.DataFrame
        DataFrame with added transaction feature columns
    """
    result = df.copy()

    # Extract month from tran_date
    if not pd.api.types.is_datetime64_any_dtype(result['tran_date']):
        result['tran_date'] = pd.to_datetime(result['tran_date'], errors='coerce')

    month = result['tran_date'].dt.month

    result['q1_trans'] = month.isin([1, 2, 3]).astype(int)
    result['q2_trans'] = month.isin([4, 5, 6]).astype(int)
    result['q3_trans'] = month.isin([7, 8, 9]).astype(int)
    result['q4_trans'] = month.isin([10, 11, 12]).astype(int)

    tran_amt = result['principal_amt'] + result['interest_amt']
    result['ck_tran_amt'] = np.where(result['acct_type'] == 'CK', tran_amt, 0)
    result['sv_tran_amt'] = np.where(result['acct_type'] == 'SV', tran_amt, 0)
    result['cc_tran_amt'] = np.where(result['acct_type'] == 'CC', tran_amt, 0)

    return result


def build_ads(customer_df, accounts_df, transactions_df):
    """
    Build the full Analytic Data Set (ADS) from the three source tables.

    Performs left joins and aggregation by cust_id to produce the 28-column ADS.

    Parameters
    ----------
    customer_df : pd.DataFrame
        Customer table
    accounts_df : pd.DataFrame
        Accounts table
    transactions_df : pd.DataFrame
        Transactions table

    Returns
    -------
    pd.DataFrame
        ADS with 28 columns, one row per customer
    """
    # Create features
    cust = create_customer_features(customer_df)
    acct = create_account_features(accounts_df)

    # Join accounts and transactions
    acct_trans = acct.merge(transactions_df, on='acct_nbr', how='left')
    acct_trans = create_transaction_features(acct_trans)

    # Join with customer
    joined = cust.merge(acct_trans, on='cust_id', how='left')

    # Replace NaN with 0 for indicator and amount columns
    fill_cols = [
        'ck_acct_ind', 'sv_acct_ind', 'cc_acct_ind',
        'ck_bal', 'sv_bal', 'cc_bal',
        'ck_tran_amt', 'sv_tran_amt', 'cc_tran_amt',
        'q1_trans', 'q2_trans', 'q3_trans', 'q4_trans',
    ]
    for col in fill_cols:
        if col in joined.columns:
            joined[col] = joined[col].fillna(0)

    # Aggregate by cust_id
    agg_dict = {
        'income': 'min',
        'age': 'min',
        'years_with_bank': 'min',
        'nbr_children': 'min',
        'female_ind': 'min',
        'single_ind': 'min',
        'married_ind': 'min',
        'separated_ind': 'min',
        'ca_resident_ind': 'max',
        'ny_resident_ind': 'max',
        'tx_resident_ind': 'max',
        'il_resident_ind': 'max',
        'az_resident_ind': 'max',
        'oh_resident_ind': 'max',
        'ck_acct_ind': 'max',
        'sv_acct_ind': 'max',
        'cc_acct_ind': 'max',
        'ck_bal': 'mean',
        'sv_bal': 'mean',
        'cc_bal': 'mean',
        'ck_tran_amt': 'mean',
        'sv_tran_amt': 'mean',
        'cc_tran_amt': 'mean',
        'q1_trans': 'sum',
        'q2_trans': 'sum',
        'q3_trans': 'sum',
        'q4_trans': 'sum',
    }

    ads = joined.groupby('cust_id').agg(agg_dict).reset_index()

    # Rename columns to final ADS names
    ads = ads.rename(columns={
        'income': 'tot_income',
        'age': 'tot_age',
        'years_with_bank': 'tot_cust_years',
        'nbr_children': 'tot_children',
        'ck_bal': 'ck_avg_bal',
        'sv_bal': 'sv_avg_bal',
        'cc_bal': 'cc_avg_bal',
        'ck_tran_amt': 'ck_avg_tran_amt',
        'sv_tran_amt': 'sv_avg_tran_amt',
        'cc_tran_amt': 'cc_avg_tran_amt',
        'q1_trans': 'q1_trans_cnt',
        'q2_trans': 'q2_trans_cnt',
        'q3_trans': 'q3_trans_cnt',
        'q4_trans': 'q4_trans_cnt',
    })

    # Fill any remaining NaN with 0
    ads = ads.fillna(0)

    return ads
