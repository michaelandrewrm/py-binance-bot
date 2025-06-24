# py-binance-bot

AI-powered Binance trading bot with Python

```bash
binance_bot/
  │
  ├── main.py          # Entry point to run the bot and orchestrate everything
  ├── visualizer.py    # Contains code to visualize data and predictions
  ├── model.py         # Contains the AI model (LSTM)
  ├── data_loader.py   # Contains functions to load historical data from Binance API
  ├── config.py        # Contains constants like SYMBOL, INTERVAL, API keys
  └── requirements.txt # Python dependencies (already provided)
```

# 📦 Trading Project Summary – Detailed Report

---

## 1. Data Pipeline & Preprocessing

### 🔹 Historical Data Loader
- Implemented `get_historical_klines()` to retrieve OHLCV data for any symbol and interval.
- Designed for flexible use with Binance or other APIs.

### 🔹 Feature Engineering
- Built a `FeatureEngineer` class to:
  - Normalize features using `StandardScaler`.
  - Add rolling technical indicators.
  - Apply `.transform_live()` for real-time prediction.
  - Save/load scalers via `scaler.pkl`.

### 🔹 Label Creation
- Binary labels indicating if next price > current price.
- Prepared for classification tasks.

---

## 2. Modeling & Training

### 🔹 Model Architecture
- LSTM-based sequential model:
  - Tunable layers: `units1`, `units2`, `dropout1`, `dropout2`, `dense_units`.
  - Binary classification output.

### 🔹 Training Strategy
- Used `TimeSeriesSplit` (5-fold) to preserve sequence ordering.
- Balanced data per fold using upsampling.

### 🔹 Hyperparameter Tuning
- Used `Keras Tuner (Hyperband)`:
  - Optimized architecture & learning rate.
  - Saved best model per symbol.

### 🔹 Final Model
- After CV, retrained best model on full data.
- Saved final model as `final_model_full_data.keras`.

---

## 3. Trading Logic

### 🔹 Prediction Logic (`predict_and_trade()`)
- Predicts next move using last `window` data.
- Applies smoothing (e.g., 3-step moving average).
- Signals:
  - `> 0.6` → BUY
  - `< 0.4` → SELL
  - Else → HOLD

### 🔹 Order Execution
- In simulation: prints intended trades.
- In live: uses portfolio logic to execute.

---

## 4. Portfolio Management

### 🔹 Features
- Tracks:
  - `cash`
  - `positions` (per symbol)
  - `avg_price`
  - `trade_log`

### 🔹 Persistence
- Loads/saves state from JSON files: `portfolio_<symbol>.json`.
- Ensures resumption across sessions.

### 🔹 Execution
- Updates state on buy/sell.
- Prevents overbuying/selling.
- Calculates average prices and logs trades.

### 🔹 Reporting
- `summary()` prints:
  - Cash
  - Positions
  - Unrealized PnL per symbol

---

## 5. Evaluation & Visualization

### 🔹 Evaluation
- Reports:
  - Accuracy
  - Loss
  - Per-fold and final metrics

### 🔹 Visualization (planned)
- `plot_prediction_vs_price()` (to be integrated):
  - Visualizes probabilities vs. prices.
  - Useful for backtesting/trade validation.

---

## 6. Learning Outcomes

| Topic | Learning |
|-------|----------|
| Time Series | Preserving order & windowing is critical |
| Balancing | Used upsampling to balance labels |
| Hyperparameter Search | Used Keras Tuner for automatic tuning |
| Portfolio Simulation | Built realistic trade logic with position tracking |
| State Persistence | Saved portfolio to disk for continuity |
| Modularization | Structured project for reuse and extension |

---

## 7. Next Steps

### 🧪 Backtesting
- Simulate over full historical dataset.

### 📈 Metrics
- Add Precision, Recall, F1, Sharpe, Drawdown.

### 📊 Visualization
- Complete `plot_prediction_vs_price()` with trade markers.

### 🌐 Live Integration
- Connect to Binance/ccxt for real trades.

### 🛠️ CLI Interface
- Add flags/configs for tuning, symbol, mode.

### 📁 Project Refactor
```bash
trading_model/
├── models/
├── data/
├── strategies/
├── portfolio/
└── main.py
```

---

Let me know if you'd like a PDF export or GitHub README format!
