import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, GRU, BatchNormalization, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

def build_model(hp):
    """
    Builds a sequential Keras model with tunable hyperparameters for GRU layers, dense layers, dropout rates, 
    activation functions, L2 regularization, and learning rate. The model uses bidirectional GRU layers followed 
    by batch normalization, dropout, and dense layers. Hyperparameters are tuned using the `hp` object.
    
    Args:
        hp: HyperParameters object for tuning model parameters.
    
    Returns:
        Compiled Keras model.
    """
        
    model = Sequential()
    model.add(Input(shape=(None, 11)))

    dropout_base = hp.Float("dropout_base", 0.1, 0.5, step=0.1)

    for i in range(hp.Int("num_gru_layers", 1, 2)):  # Optionally add 1 or 2 layers
        model.add(Bidirectional(GRU(
            units=hp.Int(f"units_{i}", min_value=32, max_value=256, step=32),
            return_sequences=(i < hp.get("num_gru_layers") - 1),
            kernel_regularizer=l2(hp.Choice(f"l2_reg_{i}", values=[1e-4, 1e-5])),
            kernel_initializer="he_uniform"
        )))

    model.add(BatchNormalization())
    model.add(Dropout(dropout_base * 1.0))

    activation = hp.Choice("activation", values=["relu", "tanh", "elu"])
    model.add(Dense(
        units=hp.Int("dense_units", min_value=8, max_value=64, step=8),
        activation=activation
    ))
    model.add(Dropout(hp.Float("dropout_dense", 0.0, 0.3, step=0.1)))
    model.add(Dense(1, activation='linear'))

    lr = hp.Choice("learning_rate", values=[1e-2, 1e-3, 5e-4, 1e-4])
    model.compile(optimizer=Adam(learning_rate=lr), loss=Huber(delta=1.0))

    return model