from dotenv import load_dotenv
import os

# load environment variables from .env file
load_dotenv()  

# Binance API keys
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://testnet.binance.vision/api"
USE_TESTNET = True

# Trading configurations
SYMBOL = "WLDUSDC"
INTERVAL = "1h"
WINDOW = 30
QUANTITY = 0.001
START_DATE = "1 year ago UTC"

# LSTM model configurations
N_STEPS = 60              # Number of past time steps for LSTM input
BATCH_SIZE = 32           # Batch size for model training
EPOCHS = 30               # Number of epochs for training the model
