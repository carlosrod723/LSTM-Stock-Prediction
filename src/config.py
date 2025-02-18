# src/config.py
DATA_PATH = "./data/model_raw_data_short.csv"

# Training hyperparameters
BATCH_SIZE = 64
EPOCHS = 50              
LEARNING_RATE = 0.0003   # Lower learning rate for finer adjustments

# Model architecture hyperparameters
LSTM_UNITS_1 = 128       # Units in first LSTM layer
LSTM_UNITS_2 = 64        # Units in second LSTM layer
DROPOUT_RATE = 0.3       # Dropout rate for better regularization

# Prediction settings
PREDICTION_HORIZON = 7   # days ahead

# Early stopping configuration
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001
