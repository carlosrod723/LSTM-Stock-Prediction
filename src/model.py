# src/model.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from src.config import LSTM_UNITS_1, LSTM_UNITS_2, DROPOUT_RATE, LEARNING_RATE

def build_lstm_model(input_shape):
    """
    Build and compile an LSTM model with three LSTM layers and an additional Dense layer.
    
    The model is designed to predict the normalized log-transformed close price.
    
    Args:
        input_shape (tuple): Shape of input sequences (sequence_length, num_features)
        
    Returns:
        model (tf.keras.Model): A compiled Keras model.
    """
    model = Sequential()
    # First LSTM layer
    model.add(LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(DROPOUT_RATE))
    
    # Second LSTM layer
    model.add(LSTM(LSTM_UNITS_2, return_sequences=True))
    model.add(Dropout(DROPOUT_RATE))
    
    # Third LSTM layer
    model.add(LSTM(32))
    model.add(Dropout(DROPOUT_RATE))
    
    # Additional Dense layer to boost capacity
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='mse',
                  metrics=['mae'])
    return model
