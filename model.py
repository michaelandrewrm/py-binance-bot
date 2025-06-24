from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_model(hp):
    # model = Sequential([
    #     Input(shape=input_shape),
    #     LSTM(
    #         units=hp.Int("units1", 64, 256, step=64),
    #         return_sequences=True,
    #         kernel_regularizer=l2(1e-4)
    #     ),
    #     BatchNormalization(),
    #     Dropout(rate=hp.Float("dropout1", 0.0, 0.5, step=0.1)),
    #     LSTM(
    #         units=hp.Int("units2", 32, 128, step=32),
    #         kernel_regularizer=l2(1e-4)
    #     ),
    #     BatchNormalization(),
    #     Dropout(rate=hp.Float("dropout2", 0.0, 0.5, step=0.1)),
    #     Dense(
    #         units=hp.Int("dense_units", 16, 64, step=16),
    #         activation="relu",
    #         kernel_regularizer=l2(1e-4)
    #     ),
    #     Dense(1, activation="sigmoid")
    # ])

    # learning_rate = hp.Float("learning_rate", 1e-4, 1e-3, sampling="log")
    # model.compile(
    #     optimizer=Adam(learning_rate=learning_rate),
    #     loss="binary_crossentropy",
    #     metrics=["accuracy"]
    # )

    # Build the LSTM model
    model = Sequential()

    # Input layer - specify the input shape here
    # model.add(Input(shape=(X_train.shape[1], 1)))
    model.add(Input(shape=(60, 1)))

    # LSTM layer
    # model.add(LSTM(
    #     units=hp.Int('units', min_value=50, max_value=200, step=50),
    #     return_sequences=True))
    
    # # Add second LSTM layer
    # model.add(LSTM(
    #     units=hp.Int('units', min_value=50, max_value=200, step=50),
    #     return_sequences=False))  # This one will return the final output
    
    # Bidirectional LSTM
    model.add(Bidirectional(LSTM(
        units=hp.Int('units', min_value=50, max_value=200, step=50),
        return_sequences=False)))  # Bidirectional LSTM

    # Dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Dense output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model
