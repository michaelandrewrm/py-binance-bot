from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_model(hp, input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(
            units=hp.Int('units1', 64, 256, step=64),
            return_sequences=True,
            kernel_regularizer=l2(1e-4)
        ),
        BatchNormalization(),
        Dropout(rate=hp.Float('dropout1', 0.0, 0.5, step=0.1)),
        LSTM(
            units=hp.Int('units2', 32, 128, step=32),
            kernel_regularizer=l2(1e-4)
        ),
        BatchNormalization(),
        Dropout(rate=hp.Float('dropout2', 0.0, 0.5, step=0.1)),
        Dense(
            units=hp.Int('dense_units', 16, 64, step=16),
            activation="relu",
            kernel_regularizer=l2(1e-4)
        ),
        Dense(1, activation="sigmoid")
    ])

    learning_rate = hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model
