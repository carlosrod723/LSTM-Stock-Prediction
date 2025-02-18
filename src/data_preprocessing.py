# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.config import DATA_PATH, PREDICTION_HORIZON

def load_data(nrows=None):
    """
    Load the dataset from the CSV file.
    Optionally, load only the first nrows rows.
    """
    df = pd.read_csv(DATA_PATH, nrows=nrows)
    df['dateDailyStockValue'] = pd.to_datetime(df['dateDailyStockValue'])
    df.sort_values(['symbolDailyStockValue', 'dateDailyStockValue'], inplace=True)
    return df

def preprocess_data(df):
    """
    Preprocess the dataset:
    - Fill missing values.
    - Select key columns while preserving the date.
    - Convert symbol to numerical codes.
    - Apply a log transform to the target ('closeDailyStockValue') and store it as 'log_close'.
    - Normalize numeric features (except date and symbol) and the target.
    
    Returns:
        df: Preprocessed DataFrame.
        feature_scaler: Fitted scaler for feature columns.
        target_scaler: Fitted scaler for the target (log_close).
    """
    # Fill missing values
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Select key columns
    key_columns = [
        'dateDailyStockValue',
        'symbolDailyStockValue',
        'openDailyStockValue',
        'highDailyStockValue',
        'lowDailyStockValue',
        'closeDailyStockValue',
        'volumeDailyStockValue'
    ]
    df = df[key_columns]
    
    # Convert symbol to numerical codes
    df['symbolDailyStockValue'] = df['symbolDailyStockValue'].astype('category').cat.codes
    
    # Apply log transform to the target price
    df['log_close'] = np.log1p(df['closeDailyStockValue'])
    # Optionally, you could drop the original closeDailyStockValue if no longer needed:
    # df.drop('closeDailyStockValue', axis=1, inplace=True)
    
    # Normalize numeric feature columns (except date and symbol)
    feature_cols = ['openDailyStockValue', 'highDailyStockValue', 'lowDailyStockValue', 'volumeDailyStockValue']
    feature_scaler = MinMaxScaler()
    df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    
    # Normalize target: log_close
    target_scaler = MinMaxScaler()
    df[['log_close']] = target_scaler.fit_transform(df[['log_close']])
    
    return df, feature_scaler, target_scaler

def create_sequences(df, sequence_length=30):
    """
    Create sequences for the LSTM model.
    Each sequence has length `sequence_length` and the target is the 'log_close'
    value PREDICTION_HORIZON days after the end of the sequence.
    
    Returns:
        X: Array of shape (num_samples, sequence_length, num_features)
        y: Array of shape (num_samples,)
        indices: Array of row indices from df corresponding to the target day for each sequence.
    """
    target_col = 'log_close'
    # Use all columns except the date and target as features.
    feature_cols = [col for col in df.columns if col not in ["dateDailyStockValue", target_col]]
    
    X, y, indices = [], [], []
    grouped = df.groupby('symbolDailyStockValue')
    for symbol, group in grouped:
        group = group.sort_index()  # Ensures chronological order within each stock.
        group_values = group[feature_cols].values
        target_values = group[target_col].values
        group_indices = group.index.values
        for i in range(len(group_values) - sequence_length - PREDICTION_HORIZON + 1):
            X.append(group_values[i: i + sequence_length])
            y.append(target_values[i + sequence_length + PREDICTION_HORIZON - 1])
            indices.append(group_indices[i + sequence_length + PREDICTION_HORIZON - 1])
    return np.array(X), np.array(y), np.array(indices)
