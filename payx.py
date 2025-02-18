# payx.py
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Import visualization functions from utils.py
from src.utils import (
    plot_predictions, 
    evaluate_relative_error, 
    calculate_mape, 
    plot_scatter_actual_vs_predicted, 
    plot_relative_error_distribution
)

def preprocess_payx(df):
    """
    Preprocess PAYX data (short-date-range version):
    - Fill missing values.
    - Select key columns.
    - Convert dateDailyStockValue to datetime.
    - Encode symbolDailyStockValue as numeric.
    - Sort by date.
    - Compute log_close as the log-transformed closeDailyStockValue.
    - Normalize feature columns and log_close.
    
    Returns:
        df: Preprocessed DataFrame.
        feature_scaler: Fitted scaler for feature columns.
        target_scaler: Fitted scaler for target (log_close).
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
    
    # Convert dateDailyStockValue to datetime and sort
    df['dateDailyStockValue'] = pd.to_datetime(df['dateDailyStockValue'], errors='coerce')
    df.sort_values('dateDailyStockValue', inplace=True)
    
    # Encode symbolDailyStockValue to numeric (for PAYX it will be constant)
    df['symbolDailyStockValue'] = df['symbolDailyStockValue'].astype('category').cat.codes
    
    # Compute log transform of closeDailyStockValue and create log_close column
    df['log_close'] = np.log1p(df['closeDailyStockValue'])
    
    # Normalize feature columns (all columns except dateDailyStockValue and log_close)
    feature_cols = [col for col in df.columns if col not in ['dateDailyStockValue', 'log_close']]
    from sklearn.preprocessing import MinMaxScaler
    feature_scaler = MinMaxScaler()
    df.loc[:, feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    
    # Normalize target: log_close
    target_scaler = MinMaxScaler()
    df.loc[:, ['log_close']] = target_scaler.fit_transform(df[['log_close']])
    
    return df, feature_scaler, target_scaler

def create_sequences_payx(df, sequence_length=30, prediction_horizon=7):
    """
    Create sequences for PAYX data.
    
    Each sequence is of length `sequence_length` and the target is the log_close value
    prediction_horizon days after the end of the sequence.
    
    Returns:
        X: numpy array of shape (num_samples, sequence_length, num_features)
        y: numpy array of shape (num_samples,)
        indices: numpy array of row indices corresponding to the target day.
    """
    target_col = 'log_close'
    # Use all columns except 'dateDailyStockValue' and target.
    feature_cols = [col for col in df.columns if col not in ['dateDailyStockValue', target_col]]
    
    X, y, indices = [], [], []
    df = df.sort_values('dateDailyStockValue')
    
    # Ensure data is numeric (float32)
    data = df[feature_cols].values.astype('float32')
    target = df[target_col].values.astype('float32')
    indices_arr = df.index.values
    
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(target[i+sequence_length+prediction_horizon - 1])
        indices.append(indices_arr[i+sequence_length+prediction_horizon - 1])
    
    return np.array(X), np.array(y), np.array(indices)

def main():
    print("NOTE: This script visualizes PAYX predictions using data from 2010-01-04 to 2010-11-29 (the training range).")
    
    # Load cleaned data from the "data" folder
    df_raw = pd.read_csv("./data/cleaned_stock_data.csv")
    
    # Convert dateDailyStockValue to datetime
    df_raw['dateDailyStockValue'] = pd.to_datetime(df_raw['dateDailyStockValue'], errors='coerce')
    print("Raw date range in cleaned dataset:", df_raw['dateDailyStockValue'].min(), "to", df_raw['dateDailyStockValue'].max())
    
    # Filter the dataset to only include dates between 2010-01-04 and 2010-11-29
    start_filter = pd.to_datetime("2010-01-04")
    end_filter = pd.to_datetime("2010-11-29")
    df_filtered = df_raw[(df_raw['dateDailyStockValue'] >= start_filter) & (df_raw['dateDailyStockValue'] <= end_filter)]
    print("Filtered date range:", df_filtered['dateDailyStockValue'].min(), "to", df_filtered['dateDailyStockValue'].max())
    
    # Filter data for PAYX rows
    df_payx = df_filtered[df_filtered['symbolDailyStockValue'] == "PAYX"].copy()
    if df_payx.empty:
        print("No data found for PAYX in the specified date range.")
        return
    
    print(f"PAYX data spans from {df_payx['dateDailyStockValue'].min()} to {df_payx['dateDailyStockValue'].max()}")
    
    # Preprocess the PAYX data
    df_payx, feature_scaler, target_scaler = preprocess_payx(df_payx)
    
    # Create sequences for PAYX
    sequence_length = 30
    X_payx, y_payx, indices_payx = create_sequences_payx(df_payx, sequence_length=sequence_length, prediction_horizon=7)
    print(f"PAYX sequences created. X shape: {X_payx.shape}, y shape: {y_payx.shape}")
    
    # Load the pre-trained model
    model = load_model("lstm_model.h5")
    
    # Generate predictions on the PAYX sequences
    y_pred = model.predict(X_payx)
    
    # Inverse-transform predictions and actual targets to original scale (log domain)
    y_payx_inv = target_scaler.inverse_transform(y_payx.reshape(-1, 1)).flatten()
    y_pred_inv = target_scaler.inverse_transform(y_pred).flatten()
    
    # Create a DataFrame for the results
    df_results = pd.DataFrame({
        'date': df_payx.loc[indices_payx, 'dateDailyStockValue'],
        'actual_inv': y_payx_inv,
        'predicted_inv': y_pred_inv
    })
    df_results.sort_values('date', inplace=True)
    
    # Evaluate and plot the results
    df_results = evaluate_relative_error(df_results)
    calculate_mape(df_results, actual_col='actual_inv', predicted_col='predicted_inv')
    
    plot_predictions(
        df_results,
        title="PAYX Predictions vs Actual Prices (2010-01-04 to 2010-11-29)",
        save_fig=True,
        filename="predictions_chart_PAYX_short_range.png"
    )
    
    plot_scatter_actual_vs_predicted(
        df_results, 
        save_fig=True, 
        filename="scatter_actual_vs_predicted_PAYX_short_range.png"
    )
    
    plot_relative_error_distribution(
        df_results, 
        save_fig=True, 
        filename="relative_error_distribution_PAYX_short_range.png"
    )
    
if __name__ == '__main__':
    main()
