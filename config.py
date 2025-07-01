import os
from dotenv import load_dotenv
from datetime import timedelta

# load environment variables from .env file
load_dotenv()  

# Binance API keys
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://testnet.binance.vision/api"
USE_TESTNET = True

# Trading configurations
SYMBOL = "BTCUSDC"
QUANTITY = 0.001
START_DATE = "1 year ago UTC"

def get_config_for_interval(interval):
    INTERVAL_CONFIG = {
        "5m": {
            "model": {
                "N_STEPS": 60,
                "EPOCHS": 30,
                "BATCH_SIZE": 32
            },
            "grid": {
                "grid_spacing": 0.0035,
                "levels": 2,
                "quantity": 100,
                "volatility_window": 10,
                "min_volatility_threshold": 0.002,
                "min_prediction_change": 0.001,
                "min_expected_gain": 0.0015,
                "cooldown_period": timedelta(minutes=15),
                "max_trades_per_day": 12,
                "max_daily_loss": 150,
                "trailing_trigger_factor": 0.5,
                "position_timeout_minutes": 15
                # "base_take_profit_pct": 0.001,
                # "base_stop_loss_pct": 0.0015,
                # "trailing_buffer_pct": 0.005,
                # "entry_multiplier": 1.0,
                # "min_price_delta": 0.001,
            },
        },
        "15m": {
            "model": {
                "N_STEPS": 96,
                "EPOCHS": 40,
                "BATCH_SIZE": 32,
            },
            "grid": {

                "grid_spacing": 0.005,
                "levels": 2,
                "quantity": 100,
                "base_take_profit_pct": 0.012,
                "base_stop_loss_pct": 0.01,
                "min_prediction_change": 0.0025,
                "volatility_window": 14,
                "cooldown_period": timedelta(minutes=45),
                "max_daily_loss": 200,
                "max_trades_per_day": 6,
                "min_volatility_threshold": 0.002,
                "trailing_trigger_factor": 0.5,
                "trailing_buffer_pct": 0.005,
                "entry_multiplier": 1.2
            },
        },
    }

    if interval not in INTERVAL_CONFIG:
        print(f"[WARN] Interval {interval} not found. Falling back to default config.")

    return INTERVAL_CONFIG.get(interval, INTERVAL_CONFIG["5m"])
