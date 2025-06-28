from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_model(hp):
    # Build the LSTM model
    model = Sequential()

    # Input layer - specify the input shape here
    model.add(Input(shape=(None, 5)))

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
        units=hp.Int('units', min_value=32, max_value=256, step=32),
        return_sequences=False)))  # Bidirectional LSTM

    # Dropout to prevent overfitting
    model.add(Dropout(0.2))

    # Dense output layer
    model.add(Dense(units=1))

    # Compile the model
    model.compile(optimizer=Adam(), loss='mean_squared_error')
    
    return model
