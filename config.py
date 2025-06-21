from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env file

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://testnet.binance.vision/api"
SYMBOL = "WLDUSDC"
INTERVAL = "1h"
WINDOW = 30
QUANTITY = 0.001
LOOKBACK = "1 year ago UTC"
