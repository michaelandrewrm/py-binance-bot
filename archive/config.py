import os
from dotenv import load_dotenv
from datetime import timedelta

load_dotenv()  

# Binance API keys
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

BASE_URL = "https://testnet.binance.vision/api"
TESTNET = True

SYMBOL = "BTCUSDC"
START_DATE = "1 month ago UTC"

SYMBOLS = {
    "BTCUSDC": {
        "base_asset": "BTC",
        "quote_asset": "USDC",
        "tick_size": 0.1,
        "step_size": 0.00001,
        "min_qty": 0.0001,
        "min_notional": 10,
        "atr_multiplier": 2.0,
        "default_grid_levels": 7,
        "default_grid_range_pct": 0.06,
        "investment_amount": 1000,
        "model": {
            "N_STEPS": 60,
            "EPOCHS": 30,
            "BATCH_SIZE": 32,
            "MAX_TRIALS": 50,
            "VALIDATION_SPLIT": 0.1,
            "LOSS": "huber",
            "EARLY_STOPPING_PATIENCE": 5,
            "LEARNING_RATES": [1e-2, 1e-3, 5e-4, 1e-4],
            "DROPOUT_RATES": [0.2, 0.3, 0.4],
            "GRU_UNITS": [64, 128, 256],
            "L2_REG": [1e-4, 1e-5],
        }
    },
    "ETHUSDC": {
        "base_asset": "ETH",
        "quote_asset": "USDC",
        "tick_size": 0.01,
        "step_size": 0.0001,
        "min_qty": 0.001,
        "min_notional": 10,
        "atr_multiplier": 2.0,
        "default_grid_levels": 9,
        "default_grid_range_pct": 0.08,
        "investment_amount": 500,
        "model": {
            "N_STEPS": 60,
            "EPOCHS": 30,
            "BATCH_SIZE": 32,
            "MAX_TRIALS": 50,
            "VALIDATION_SPLIT": 0.1,
            "LOSS": "huber",
            "EARLY_STOPPING_PATIENCE": 5,
            "LEARNING_RATES": [1e-2, 1e-3, 5e-4, 1e-4],
            "DROPOUT_RATES": [0.2, 0.3, 0.4],
            "GRU_UNITS": [64, 128, 256],
            "L2_REG": [1e-4, 1e-5],
        }
    },
    "WLDUSDC": {
        "base_asset": "WLD",
        "quote_asset": "USDC",
        "tick_size": 0.001,
        "step_size": 0.001,
        "min_qty": 0.1,
        "min_notional": 10,
        "atr_multiplier": 2.5,
        "default_grid_levels": 10,
        "default_grid_range_pct": 0.15,
        "investment_amount": 200,
        "model": {
            "N_STEPS": 60,
            "EPOCHS": 35,
            "BATCH_SIZE": 32,
            "MAX_TRIALS": 60,
            "VALIDATION_SPLIT": 0.1,
            "LOSS": "huber",
            "EARLY_STOPPING_PATIENCE": 6,
            "LEARNING_RATES": [1e-2, 1e-3, 5e-4, 1e-4],
            "DROPOUT_RATES": [0.2, 0.3, 0.4],
            "GRU_UNITS": [64, 128, 256],
            "L2_REG": [1e-4, 1e-5],
        },
    }
}

def get_config_for_symbol(symbol):
    if symbol not in SYMBOLS:
        raise ValueError(f"Unknown symbol '{symbol}'! Choose from: {list(SYMBOLS.keys())}")
    return SYMBOLS[symbol]

def get_config_for_interval(interval, symbol):
    DEFAULT_GRID_CONFIG = {
        "grid": {
            "lower_price": 0,
            "upper_price": 0,
            "levels": 0,
            "quantity": 0,
            "fee": 0.001,
        },
    }
    SYMBOL_CONFIG = get_config_for_symbol(symbol)
    SYMBOL_CONFIG.update(DEFAULT_GRID_CONFIG)
    return SYMBOL_CONFIG
