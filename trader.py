import os
import joblib
import numpy as np
import pandas as pd
from binance.client import Client
from visualizer import (
    plot_trade_history,
    plot_prediction_distribution,
    plot_predictions_vs_price_live
)

class TradeSimulator:
    def __init__(self, model, platt_model, window, feature_engineer, symbol=None, trade_quantity=1):
        self.model = model
        self.platt_model = platt_model
        self.window = window
        self.feature_engineer = feature_engineer
        self.symbol = symbol
        self.trade_quantity = trade_quantity
        self.pred_history = []
        self.prediction_log = []
        self.trade_history = []

        # Initialize Binance client for live trading
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = Client(api_key, api_secret) if api_key and api_secret else None

    def dry_run_warning(self):
        confirm = input("\n⚠️  You are about to execute a LIVE trade. Type 'CONFIRM' to proceed: ")
        if confirm.strip().upper() != 'CONFIRM':
            print("Live trade cancelled by user.")
            return False
        return True

    def execute_buy(self, df, simulate=True):
        print(f"Executing BUY order for {self.trade_quantity} units of {self.symbol}")
        if simulate or self.client is None:
            self.trade_history.append({
                "timestamp": df.index[-1],
                "action": "BUY",
                "price": df['close'].iloc[-1],
                "quantity": self.trade_quantity
            })
        else:
            if not self.dry_run_warning():
                return
            try:
                order = self.client.order_market_buy(
                    symbol=self.symbol,
                    quantity=self.trade_quantity
                )
                print("[LIVE ORDER EXECUTED] BUY:", order)
            except Exception as e:
                print(f"[ERROR] Failed to execute live BUY order: {e}")

    def execute_sell(self, df, simulate=True):
        print(f"Executing SELL order for {self.trade_quantity} units of {self.symbol}")
        if simulate or self.client is None:
            self.trade_history.append({
                "timestamp": df.index[-1],
                "action": "SELL",
                "price": df['close'].iloc[-1],
                "quantity": self.trade_quantity
            })
        else:
            if not self.dry_run_warning():
                return
            try:
                order = self.client.order_market_sell(
                    symbol=self.symbol,
                    quantity=self.trade_quantity
                )
                print("[LIVE ORDER EXECUTED] SELL:", order)
            except Exception as e:
                print(f"[ERROR] Failed to execute live SELL order: {e}")

    def predict_and_trade(self, df, prob_threshold_buy=0.6, prob_threshold_sell=0.4,
                          smooth_window=3, simulate=True):

        df = self.feature_engineer.transform_live(df)
        feature_cols = self.feature_engineer.feature_cols

        if len(df) < self.window:
            print(f"Not enough data to make prediction (required: {self.window}, available: {len(df)})")
            return

        latest_data = df[feature_cols].iloc[-self.window:].values
        latest_data = latest_data.reshape((1, self.window, len(feature_cols)))

        raw_pred = self.model.predict(latest_data, verbose=0)[0][0]
        prediction = self.platt_model.predict_proba(np.array([[raw_pred]]))[0][1] if self.platt_model else raw_pred

        self.pred_history.append(prediction)
        if len(self.pred_history) > smooth_window:
            self.pred_history.pop(0)

        smoothed_pred = sum(self.pred_history) / len(self.pred_history)

        print(f"Raw Prediction: {raw_pred:.2f}, Calibrated: {prediction:.2f}, Smoothed: {smoothed_pred:.2f}")

        self.prediction_log.append({
            "timestamp": df.index[-1],
            "price": df['close'].iloc[-1],
            "prediction": prediction
        })

        if smoothed_pred > prob_threshold_buy:
            print("Signal: BUY")
            if simulate:
                print(f"[SIMULATION] Would execute BUY order for {self.trade_quantity} {self.symbol}")
            self.execute_buy(df, simulate=simulate)

        elif smoothed_pred < prob_threshold_sell:
            print("Signal: SELL")
            if simulate:
                print(f"[SIMULATION] Would execute SELL order for {self.trade_quantity} {self.symbol}")
            self.execute_sell(df, simulate=simulate)

        else:
            print("Signal: HOLD - no trade executed.")

        if simulate and self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            plot_trade_history(df_trades)

        if len(self.pred_history) > 1:
            plot_prediction_distribution(self.pred_history)

        if len(self.prediction_log) > 1:
            plot_predictions_vs_price_live(self.prediction_log)
