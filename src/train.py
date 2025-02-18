# src/train.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

from src.data_preprocessing import load_data, preprocess_data, create_sequences
from src.model import build_lstm_model
from src.config import BATCH_SIZE, EPOCHS, EARLY_STOPPING_PATIENCE, MIN_DELTA
from src.utils import (
    plot_training_history, 
    plot_predictions, 
    evaluate_relative_error, 
    calculate_mape, 
    plot_scatter_actual_vs_predicted, 
    plot_relative_error_distribution
)

# Custom callback to compute relative error on the validation set at the end of each epoch.
class RelativeErrorCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val, target_scaler):
        super(RelativeErrorCallback, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.target_scaler = target_scaler

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val, verbose=0)
        # Inverse-transform the normalized predictions and actual values
        y_val_inv = self.target_scaler.inverse_transform(self.y_val.reshape(-1, 1)).flatten()
        y_pred_inv = self.target_scaler.inverse_transform(y_pred).flatten()
        ratio = y_pred_inv / y_val_inv
        within_range = (ratio >= 0.7) & (ratio <= 1.3)
        percent_within_range = np.mean(within_range) * 100
        print(f"\nEpoch {epoch+1}: {percent_within_range:.2f}% of validation predictions within 70%-130% range")

def main():
    print("Loading data...")
    # Load a subset of 100,000 rows from the cleaned dataset in the data folder.
    df = load_data(nrows=100000)
    
    # Verify the timeframe of the data
    start_date = df['dateDailyStockValue'].min()
    end_date = df['dateDailyStockValue'].max()
    print(f"Data subset covers from {start_date.date()} to {end_date.date()}")
    
    print("Preprocessing data...")
    # preprocess_data returns the preprocessed df, along with feature and target scalers.
    df, feature_scaler, target_scaler = preprocess_data(df)
    
    print("Creating sequences...")
    sequence_length = 30 
    X, y, indices = create_sequences(df, sequence_length)
    print(f"Sequences created. X shape: {X.shape}, y shape: {y.shape}, indices shape: {indices.shape}")
    
    # Split the data into training and validation sets (keeping indices for merging)
    X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X, y, indices, test_size=0.2, random_state=42
    )
    
    print("Building model...")
    input_shape = X_train.shape[1:]  # (sequence_length, num_features)
    model = build_lstm_model(input_shape)
    model.summary()
    
    # Set up early stopping and the relative error callback.
    early_stopping = EarlyStopping(monitor='val_loss', 
                                   patience=EARLY_STOPPING_PATIENCE, 
                                   min_delta=MIN_DELTA, 
                                   restore_best_weights=True)
    relative_error_callback = RelativeErrorCallback(X_val, y_val, target_scaler)
    callbacks = [early_stopping, relative_error_callback]
    
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    # Save the trained model
    model.save("lstm_model.h5")
    print("Model saved as lstm_model.h5")
    
    # Plot training history
    plot_training_history(history, save_fig=True, filename="training_history.png")
    
    print("Generating predictions on the validation set...")
    y_pred = model.predict(X_val)
    
    # Inverse-transform predictions and actual values to the original scale (log domain)
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
    y_pred_inv = target_scaler.inverse_transform(y_pred).flatten()
    
    # Create a DataFrame for validation results
    df_val = pd.DataFrame({
        'row_index': idx_val,
        'actual_inv': y_val_inv,
        'predicted_inv': y_pred_inv
    })
    
    # Merge with the original df to retrieve corresponding dates and stock symbols
    df_val = df_val.merge(
        df[['dateDailyStockValue', 'symbolDailyStockValue']],
        left_on='row_index',
        right_index=True,
        how='left'
    )
    
    print("Sample of merged validation results:")
    print(df_val.head())
    
    # For visualization purposes, filter for a single stock (e.g., stock code 10)
    single_stock_code = 10  # Adjust this value as needed
    df_val_single = df_val[df_val['symbolDailyStockValue'] == single_stock_code].copy()
    df_val_single.sort_values('dateDailyStockValue', inplace=True)
    
    # Prepare DataFrame for plotting and further evaluation
    df_results = pd.DataFrame({
        'date': df_val_single['dateDailyStockValue'],
        'actual_inv': df_val_single['actual_inv'],
        'predicted_inv': df_val_single['predicted_inv']
    })
    
    # Evaluate the relative error for this single stock
    # IMPORTANT: capture the returned DataFrame so it has 'relative_ratio'
    df_results = evaluate_relative_error(df_results)
    
    # Calculate and print the Mean Absolute Percentage Error (MAPE)
    calculate_mape(df_results, actual_col='actual_inv', predicted_col='predicted_inv')
    
    # Generate and save the predictions chart (time series)
    plot_predictions(
        df_results,
        title=f"Predictions for Stock Symbol {single_stock_code}",
        save_fig=True,
        filename="predictions_chart_single_stock.png"
    )
    
    # Plot a scatter plot of actual vs. predicted values
    plot_scatter_actual_vs_predicted(df_results, save_fig=True, filename="scatter_actual_vs_predicted.png")
    
    # Plot the distribution of relative error ratios (now that df_results has 'relative_ratio')
    plot_relative_error_distribution(df_results, save_fig=True, filename="relative_error_distribution.png")

if __name__ == '__main__':
    main()
