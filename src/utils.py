# src/utils.py
import os
import matplotlib.pyplot as plt

def plot_training_history(history, save_fig=False, filename="training_history.png"):
    """
    Plot training and validation loss and MAE.
    
    Args:
        history: Keras History object from model.fit().
        save_fig: Boolean, if True the plot will be saved.
        filename: Name of the file to save the plot.
    """
    plt.figure(figsize=(14, 6))  # Increased width for better readability
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    
    if save_fig:
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        file_path = os.path.join(charts_dir, filename)
        plt.savefig(file_path)
        print(f"Chart saved to: {file_path}")
    
    plt.show()

def plot_predictions(df, title="Stock Price Predictions vs. Actual", 
                     save_fig=False, filename="predictions_chart.png"):
    """
    Plots the actual vs. predicted stock prices over time.
    
    Args:
        df (pd.DataFrame): DataFrame with columns ['date', 'actual_inv', 'predicted_inv'] 
                           (if available) or ['date', 'actual', 'predicted'].
        title (str): Chart title.
        save_fig (bool): Whether to save the figure as a file.
        filename (str): Filename for the saved chart.
    """
    plt.figure(figsize=(12, 6))  # Increased width and height
    
    # Check which columns to use
    if 'actual_inv' in df.columns and 'predicted_inv' in df.columns:
        actual_col = 'actual_inv'
        predicted_col = 'predicted_inv'
    else:
        actual_col = 'actual'
        predicted_col = 'predicted'
    
    plt.plot(df['date'], df[actual_col], marker='x', color='green', label='Actual')
    plt.plot(df['date'], df[predicted_col], marker='x', color='red', label='Predicted')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    
    if save_fig:
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        file_path = os.path.join(charts_dir, filename)
        plt.savefig(file_path)
        print(f"Chart saved to: {file_path}")
    
    plt.show()

def evaluate_relative_error(df):
    """
    Evaluate the relative error by computing the ratio of predicted to actual values.
    Uses 'actual_inv' and 'predicted_inv' if available, otherwise 'actual' and 'predicted'.
    
    Prints the percentage of predictions that fall within 70% to 130% of the actual values.
    
    Args:
        df (pd.DataFrame): DataFrame with appropriate columns.
    Returns:
        df (pd.DataFrame): The same DataFrame, now with a 'relative_ratio' column.
    """
    df = df.copy()
    if 'actual_inv' in df.columns and 'predicted_inv' in df.columns:
        actual_col = 'actual_inv'
        predicted_col = 'predicted_inv'
    else:
        actual_col = 'actual'
        predicted_col = 'predicted'
    
    df['relative_ratio'] = df[predicted_col] / df[actual_col]
    within_range = df[(df['relative_ratio'] >= 0.7) & (df['relative_ratio'] <= 1.3)]
    percentage_within_range = len(within_range) / len(df) * 100
    print(f"{percentage_within_range:.2f}% of predictions fall within 70% to 130% of the actual values.")
    return df

def calculate_mape(df, actual_col='actual_inv', predicted_col='predicted_inv'):
    """
    Calculate and print the Mean Absolute Percentage Error (MAPE).
    
    Args:
        df (pd.DataFrame): DataFrame with actual and predicted columns.
    """
    df = df.copy()
    df['ape'] = abs((df[predicted_col] - df[actual_col]) / (df[actual_col] + 1e-8))
    mape = df['ape'].mean() * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    return mape

def plot_scatter_actual_vs_predicted(df, save_fig=False, filename="scatter_actual_vs_predicted.png"):
    """
    Plots a scatter plot of actual vs. predicted values.
    
    Args:
        df (pd.DataFrame): DataFrame with columns 'actual_inv' and 'predicted_inv'
                           (or fallback to 'actual' and 'predicted').
        save_fig (bool): Whether to save the plot.
        filename (str): Filename for the saved plot.
    """
    plt.figure(figsize=(10, 8)) 
    if 'actual_inv' in df.columns and 'predicted_inv' in df.columns:
        actual_col = 'actual_inv'
        predicted_col = 'predicted_inv'
    else:
        actual_col = 'actual'
        predicted_col = 'predicted'
    
    plt.scatter(df[actual_col], df[predicted_col], alpha=0.6, color='blue', label='Predictions')
    # Plot ideal line
    min_val = min(df[actual_col].min(), df[predicted_col].min())
    max_val = max(df[actual_col].max(), df[predicted_col].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Scatter Plot of Actual vs Predicted Prices')
    plt.legend()
    plt.tight_layout()
    
    if save_fig:
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        file_path = os.path.join(charts_dir, filename)
        plt.savefig(file_path)
        print(f"Scatter plot saved to: {file_path}")
    
    plt.show()
    
def plot_relative_error_distribution(df, save_fig=False, filename="relative_error_distribution.png"):
    """
    Plot a histogram of the relative error ratios (predicted/actual).
    
    Args:
        df (pd.DataFrame): DataFrame with a 'relative_ratio' column.
        save_fig (bool): Whether to save the plot.
        filename (str): Filename for the saved plot.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['relative_ratio'], bins=50, color='skyblue', edgecolor='black')
    plt.xlabel('Relative Ratio (Predicted/Actual)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative Error Ratios')
    plt.tight_layout()
    if save_fig:
        charts_dir = os.path.join(os.path.dirname(__file__), '..', 'charts')
        os.makedirs(charts_dir, exist_ok=True)
        file_path = os.path.join(charts_dir, filename)
        plt.savefig(file_path)
        print(f"Relative error distribution saved to: {file_path}")
    plt.show()
