# model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense

def build_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),   # Updated: use Input layer here
        LSTM(50),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
