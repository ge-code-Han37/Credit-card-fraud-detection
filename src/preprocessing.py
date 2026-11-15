import pandas as pd
import numpy as np

def clean_data(df, drop_duplicates=True):
    df = df.copy()
    
    
    if drop_duplicates and df.duplicated().any():
        df = df.drop_duplicates()
        
    
    if 'amount' in df.columns:
        df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
    
    text_cols = ['merchant_category', 'payment_channel', 'home_country', 'transaction_country']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            
  
    if 'payment_channel' in df.columns:
        df['payment_channel'] = df['payment_channel'].replace({'in store': 'in-store'})
        
    country_map = {
        'us': 'usa', 'u.s.a': 'usa', 'united states': 'usa',
        'uk': 'united kingdom', 'britain': 'united kingdom'
    }
    for col in ['home_country', 'transaction_country']:
        if col in df.columns:
            df[col] = df[col].replace(country_map)

    
    if 'customer_age' in df.columns:
        df.loc[(df['customer_age'] < 18) | (df['customer_age'] > 100), 'customer_age'] = np.nan
    if 'account_age_days' in df.columns:
        df.loc[(df['account_age_days'] < 0) | (df['account_age_days'] > 10000), 'account_age_days'] = np.nan
    if 'distance_from_home_km' in df.columns:
        df.loc[(df['distance_from_home_km'] < 0) | (df['distance_from_home_km'] > 20000), 'distance_from_home_km'] = np.nan
    if 'num_txn_last_24h' in df.columns:
        df.loc[(df['num_txn_last_24h'] < 0) | (df['num_txn_last_24h'] > 50), 'num_txn_last_24h'] = np.nan
    if 'amount' in df.columns:
        df.loc[df['amount'] <= 0, 'amount'] = np.nan

  
    if 'transaction_datetime' in df.columns:
        dt = pd.to_datetime(df['transaction_datetime'], errors='coerce')
        df['hour'] = dt.dt.hour
        df['day'] = dt.dt.day
        df['month'] = dt.dt.month
        df['day_of_week'] = dt.dt.dayofweek
        
   
    if 'hour' in df.columns:
        df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(float)
    if 'amount' in df.columns and 'avg_txn_amount_30d' in df.columns:
        df['amount_ratio'] = df['amount'] / (df['avg_txn_amount_30d'] + 1e-5)
    if 'home_country' in df.columns and 'transaction_country' in df.columns:
        df['is_foreign'] = (df['home_country'] != df['transaction_country']).astype(float)

    return df