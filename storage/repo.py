"""
Repository Module - Data Persistence

This module provides data persistence functionality for the trading bot,
including database operations, file storage, and data archival.
"""

import os
import sqlite3
import json
import pickle
from typing import List, Dict, Optional, Any, Union
from decimal import Decimal
from datetime import datetime, timezone, date, timedelta
import logging
from pathlib import Path
import pandas as pd
from contextlib import contextmanager

from data.schema import KlineData, TradeRecord, BacktestResult, EquityCurvePoint, utc_isoformat
from core.state import BotState

logger = logging.getLogger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """JSON encoder that handles Decimal objects"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, datetime):
            return utc_isoformat(obj)
        if isinstance(obj, date):
            return utc_isoformat(obj) if isinstance(obj, datetime) else obj.isoformat()
        return super().default(obj)

class SQLiteRepository:
    """
    SQLite-based repository for trading data persistence
    """
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables with hardened persistence"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Enable foreign key constraints and WAL mode for better concurrency
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.execute("PRAGMA journal_mode = WAL")
            cursor.execute("PRAGMA synchronous = NORMAL")
            cursor.execute("PRAGMA cache_size = 10000")
            cursor.execute("PRAGMA temp_store = MEMORY")
            
            # Trading runs table (parent table for referential integrity)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_runs (
                    run_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_time_utc TEXT NOT NULL,
                    end_time_utc TEXT,
                    initial_capital REAL NOT NULL,
                    final_capital REAL,
                    status TEXT NOT NULL DEFAULT 'running',
                    parameters TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                )
            """)
            
            # Orders table with foreign key to trading_runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    client_order_id TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    status TEXT NOT NULL,
                    filled_quantity REAL DEFAULT 0,
                    filled_price REAL DEFAULT 0,
                    fee REAL DEFAULT 0,
                    ts_utc TEXT NOT NULL,
                    update_time_utc TEXT,
                    grid_level INTEGER,
                    strategy_tag TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    FOREIGN KEY (run_id) REFERENCES trading_runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, order_id)
                )
            """)
            
            # Fills/executions table with foreign key to orders
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS fills (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL DEFAULT 0,
                    commission_asset TEXT,
                    ts_utc TEXT NOT NULL,
                    is_maker BOOLEAN DEFAULT 0,
                    strategy_tag TEXT,
                    pnl REAL,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    FOREIGN KEY (run_id) REFERENCES trading_runs(run_id) ON DELETE CASCADE,
                    UNIQUE(run_id, trade_id)
                )
            """)
            
            # Equity curve table with foreign key to trading_runs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_curve (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    ts_utc TEXT NOT NULL,
                    total_equity REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    realized_pnl REAL DEFAULT 0,
                    cash_balance REAL NOT NULL,
                    position_value REAL DEFAULT 0,
                    drawdown REAL DEFAULT 0,
                    daily_return REAL DEFAULT 0,
                    cumulative_return REAL DEFAULT 0,
                    strategy_tag TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    FOREIGN KEY (run_id) REFERENCES trading_runs(run_id) ON DELETE CASCADE
                )
            """)
            
            # Kline data table (enhanced with UTC timestamps)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS klines (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    open_time_utc TEXT NOT NULL,
                    close_time_utc TEXT NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL DEFAULT 0,
                    trades_count INTEGER DEFAULT 0,
                    taker_buy_volume REAL DEFAULT 0,
                    taker_buy_quote_volume REAL DEFAULT 0,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    UNIQUE(symbol, timeframe, open_time_utc)
                )
            """)
            
            # Legacy trades table (maintained for backward compatibility)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    order_type TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    commission REAL,
                    commission_asset TEXT,
                    timestamp_utc TEXT NOT NULL,
                    order_id TEXT,
                    trade_id TEXT,
                    is_paper_trade BOOLEAN DEFAULT 0,
                    strategy_name TEXT,
                    pnl REAL,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                )
            """)
            
            # Backtest results table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    start_date_utc TEXT NOT NULL,
                    end_date_utc TEXT NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    total_trades INTEGER,
                    win_rate REAL,
                    parameters TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                )
            """)
            
            # Bot state table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS bot_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    state_data TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    requires_manual_resume BOOLEAN DEFAULT 0,
                    flatten_reason TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc'))
                )
            """)
            
            # Performance metrics table (enhanced)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    strategy_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    period TEXT,
                    symbol TEXT,
                    created_at_utc TEXT NOT NULL DEFAULT (datetime('now', 'utc')),
                    FOREIGN KEY (run_id) REFERENCES trading_runs(run_id) ON DELETE SET NULL
                )
            """)
            
            # Create performance indices for fast queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_run_ts ON orders(run_id, ts_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol_side ON orders(symbol, side, ts_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status, ts_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_run_ts ON fills(run_id, ts_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_symbol_side ON fills(symbol, side, ts_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fills_order_id ON fills(order_id)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_run_ts ON equity_curve(run_id, ts_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity_curve(ts_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_klines_symbol_timeframe ON klines(symbol, timeframe, open_time_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_klines_time_range ON klines(open_time_utc, close_time_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name, timestamp_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_strategy ON backtest_results(strategy_name, created_at_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_backtest_symbol ON backtest_results(symbol, start_date_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_states_time ON bot_states(timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_bot_states_manual_resume ON bot_states(requires_manual_resume, timestamp_utc)")
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_time ON performance_metrics(run_id, timestamp_utc)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_strategy_metric ON performance_metrics(strategy_name, metric_name, timestamp_utc)")
            
            conn.commit()
            logger.info("Database initialized with hardened persistence, foreign keys, and performance indices")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup and hardened settings"""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        
        # Enable hardened persistence settings
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = 10000")
        conn.execute("PRAGMA temp_store = MEMORY")
        
        try:
            yield conn
        finally:
            conn.close()
    
    def save_klines(self, klines: List[KlineData]) -> int:
        """Save kline data to database with ISO8601 UTC timestamps"""
        if not klines:
            return 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            kline_data = [
                (
                    k.symbol, k.interval, 
                    utc_isoformat(k.open_time), utc_isoformat(k.close_time),
                    float(k.open_price), float(k.high_price), 
                    float(k.low_price), float(k.close_price),
                    float(k.volume), float(k.quote_asset_volume or 0),
                    k.number_of_trades or 0,
                    float(k.taker_buy_base_volume or 0),
                    float(k.taker_buy_quote_volume or 0)
                )
                for k in klines
            ]
            
            cursor.executemany("""
                INSERT OR REPLACE INTO klines (
                    symbol, timeframe, open_time_utc, close_time_utc,
                    open_price, high_price, low_price, close_price,
                    volume, quote_volume, trades_count,
                    taker_buy_volume, taker_buy_quote_volume
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, kline_data)
            
            conn.commit()
            return cursor.rowcount
    
    def load_klines(self, symbol: str, interval: str, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   limit: Optional[int] = None) -> List[KlineData]:
        """Load kline data from database with ISO8601 UTC timestamp parsing"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM klines 
                WHERE symbol = ? AND timeframe = ?
            """
            params = [symbol, interval]
            
            if start_time:
                query += " AND open_time_utc >= ?"
                params.append(utc_isoformat(start_time))
            
            if end_time:
                query += " AND open_time_utc <= ?"
                params.append(utc_isoformat(end_time))
            
            query += " ORDER BY open_time_utc"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                KlineData(
                    symbol=row['symbol'],
                    interval=row['timeframe'],
                    open_time=datetime.fromisoformat(row['open_time_utc'].replace('Z', '+00:00')),
                    close_time=datetime.fromisoformat(row['close_time_utc'].replace('Z', '+00:00')),
                    open_price=Decimal(str(row['open_price'])),
                    high_price=Decimal(str(row['high_price'])),
                    low_price=Decimal(str(row['low_price'])),
                    close_price=Decimal(str(row['close_price'])),
                    volume=Decimal(str(row['volume'])),
                    quote_asset_volume=Decimal(str(row['quote_volume'] or 0)),
                    number_of_trades=row['trades_count'],
                    taker_buy_base_volume=Decimal(str(row['taker_buy_volume'] or 0)),
                    taker_buy_quote_volume=Decimal(str(row['taker_buy_quote_volume'] or 0))
                )
                for row in rows
            ]
    
    def save_trade(self, trade: TradeRecord) -> int:
        """Save trade record to database with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    symbol, side, order_type, quantity, price,
                    commission, commission_asset, timestamp_utc,
                    order_id, trade_id, is_paper_trade,
                    strategy_name, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade.symbol, trade.side, trade.order_type,
                float(trade.quantity), float(trade.price),
                float(trade.commission or 0), trade.commission_asset,
                utc_isoformat(trade.timestamp), trade.order_id, trade.trade_id,
                trade.is_paper_trade, trade.strategy_name,
                float(trade.pnl or 0)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def load_trades(self, symbol: Optional[str] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   strategy_name: Optional[str] = None) -> List[TradeRecord]:
        """Load trades from database with ISO8601 UTC timestamp parsing"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_time:
                query += " AND timestamp_utc >= ?"
                params.append(utc_isoformat(start_time))
            
            if end_time:
                query += " AND timestamp_utc <= ?"
                params.append(utc_isoformat(end_time))
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " ORDER BY timestamp_utc"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            return [
                TradeRecord(
                    symbol=row['symbol'],
                    side=row['side'],
                    order_type=row['order_type'],
                    quantity=Decimal(str(row['quantity'])),
                    price=Decimal(str(row['price'])),
                    commission=Decimal(str(row['commission'] or 0)),
                    commission_asset=row['commission_asset'],
                    timestamp=datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00')),
                    order_id=row['order_id'],
                    trade_id=row['trade_id'],
                    is_paper_trade=bool(row['is_paper_trade']),
                    strategy_name=row['strategy_name'],
                    pnl=Decimal(str(row['pnl'] or 0))
                )
                for row in rows
            ]
    
    def save_backtest_result(self, result: BacktestResult) -> int:
        """Save backtest result to database with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO backtest_results (
                    strategy_name, symbol, start_date_utc, end_date_utc,
                    initial_capital, final_capital, total_return,
                    sharpe_ratio, max_drawdown, total_trades,
                    win_rate, parameters
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.strategy_name, result.symbol,
                utc_isoformat(result.start_date), utc_isoformat(result.end_date),
                float(result.initial_capital), float(result.final_capital),
                float(result.total_return), float(result.metrics.sharpe_ratio),
                float(result.metrics.max_drawdown), result.metrics.total_trades,
                float(result.metrics.win_rate),
                json.dumps(result.parameters, cls=DecimalEncoder)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_bot_state(self, state: BotState) -> int:
        """Save bot state to database with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            state_json = json.dumps(state.__dict__, cls=DecimalEncoder)
            
            cursor.execute("""
                INSERT INTO bot_states (
                    state_data, timestamp_utc, requires_manual_resume, flatten_reason
                ) VALUES (?, ?, ?, ?)
            """, (
                state_json, 
                utc_isoformat(datetime.now(timezone.utc)),
                getattr(state, 'requires_manual_resume', False),
                getattr(state, 'flatten_reason', None)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def load_latest_bot_state(self) -> Optional[BotState]:
        """Load the most recent bot state with ISO8601 UTC timestamp parsing"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT state_data FROM bot_states
                ORDER BY timestamp_utc DESC LIMIT 1
            """)
            
            row = cursor.fetchone()
            if row:
                state_data = json.loads(row['state_data'])
                return BotState(**state_data)
        
        return None

    def save_trading_run(self, run_id: str, strategy_name: str, symbol: str, 
                        initial_capital: float, parameters: Dict[str, Any] = None) -> int:
        """Save trading run metadata"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO trading_runs (
                    run_id, strategy_name, symbol, start_time_utc, initial_capital, parameters
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                run_id, strategy_name, symbol,
                utc_isoformat(datetime.now(timezone.utc)),
                initial_capital,
                json.dumps(parameters or {}, cls=DecimalEncoder)
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_order(self, run_id: str, order_data: Dict[str, Any]) -> int:
        """Save order to database with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO orders (
                    run_id, order_id, client_order_id, symbol, side, order_type,
                    quantity, price, status, filled_quantity, filled_price, fee,
                    ts_utc, update_time_utc, grid_level, strategy_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                order_data['orderId'],
                order_data.get('clientOrderId'),
                order_data['symbol'],
                order_data['side'],
                order_data['type'],
                float(order_data['origQty']),
                float(order_data['price']),
                order_data['status'],
                float(order_data.get('executedQty', 0)),
                float(order_data.get('cummulativeQuoteQty', 0)) / float(order_data.get('executedQty', 1)),
                float(order_data.get('fee', 0)),
                utc_isoformat(datetime.fromtimestamp(order_data['time'] / 1000, tz=timezone.utc)),
                utc_isoformat(datetime.fromtimestamp(order_data.get('updateTime', order_data['time']) / 1000, tz=timezone.utc)),
                order_data.get('grid_level'),
                order_data.get('strategy_tag')
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_fill(self, run_id: str, fill_data: Dict[str, Any]) -> int:
        """Save fill/execution to database with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO fills (
                    run_id, order_id, trade_id, symbol, side, quantity, price,
                    commission, commission_asset, ts_utc, is_maker, strategy_tag, pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                fill_data['orderId'],
                fill_data['id'],
                fill_data['symbol'],
                fill_data['side'],
                float(fill_data['qty']),
                float(fill_data['price']),
                float(fill_data['commission']),
                fill_data['commissionAsset'],
                utc_isoformat(datetime.fromtimestamp(fill_data['time'] / 1000, tz=timezone.utc)),
                fill_data.get('isMaker', False),
                fill_data.get('strategy_tag'),
                float(fill_data.get('pnl', 0))
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_equity_point(self, run_id: str, equity_data: Dict[str, Any]) -> int:
        """Save equity curve point with ISO8601 UTC timestamps"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO equity_curve (
                    run_id, ts_utc, total_equity, unrealized_pnl, realized_pnl,
                    cash_balance, position_value, drawdown, daily_return,
                    cumulative_return, strategy_tag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                utc_isoformat(equity_data['timestamp']),
                float(equity_data['total_equity']),
                float(equity_data.get('unrealized_pnl', 0)),
                float(equity_data.get('realized_pnl', 0)),
                float(equity_data['cash_balance']),
                float(equity_data.get('position_value', 0)),
                float(equity_data.get('drawdown', 0)),
                float(equity_data.get('daily_return', 0)),
                float(equity_data.get('cumulative_return', 0)),
                equity_data.get('strategy_tag')
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def load_orders(self, run_id: str, symbol: Optional[str] = None, 
                   status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load orders for a trading run"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM orders WHERE run_id = ?"
            params = [run_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            query += " ORDER BY ts_utc"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def load_fills(self, run_id: str, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load fills for a trading run"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM fills WHERE run_id = ?"
            params = [run_id]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY ts_utc"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def load_equity_curve(self, run_id: str, 
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Load equity curve for a trading run"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM equity_curve WHERE run_id = ?"
            params = [run_id]
            
            if start_time:
                query += " AND ts_utc >= ?"
                params.append(utc_isoformat(start_time))
            
            if end_time:
                query += " AND ts_utc <= ?"
                params.append(utc_isoformat(end_time))
            
            query += " ORDER BY ts_utc"
            
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_trading_run_status(self, run_id: str, status: str, final_capital: Optional[float] = None):
        """Update trading run status and final capital"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if final_capital is not None:
                cursor.execute("""
                    UPDATE trading_runs 
                    SET status = ?, final_capital = ?, end_time_utc = ?
                    WHERE run_id = ?
                """, (status, final_capital, utc_isoformat(datetime.now(timezone.utc)), run_id))
            else:
                cursor.execute("""
                    UPDATE trading_runs 
                    SET status = ?
                    WHERE run_id = ?
                """, (status, run_id))
            
            conn.commit()
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Clean up old data based on retention policy"""
        cutoff_date = utc_isoformat(datetime.now(timezone.utc) - timedelta(days=days_to_keep))
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Clean old klines
            cursor.execute("DELETE FROM klines WHERE created_at_utc < ?", (cutoff_date,))
            klines_deleted = cursor.rowcount
            
            # Clean old bot states  
            cursor.execute("DELETE FROM bot_states WHERE created_at_utc < ?", (cutoff_date,))
            states_deleted = cursor.rowcount
            
            # Clean old completed trading runs and cascade to orders/fills/equity
            cursor.execute("DELETE FROM trading_runs WHERE end_time_utc < ? AND status = 'completed'", (cutoff_date,))
            runs_deleted = cursor.rowcount
            
            conn.commit()
            
        logger.info(f"Cleaned up {klines_deleted} klines, {states_deleted} bot states, {runs_deleted} trading runs")

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health metrics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            stats = {}
            
            # Table row counts
            tables = ['trading_runs', 'orders', 'fills', 'equity_curve', 'klines', 'trades', 'bot_states']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            stats['db_size_mb'] = (page_count * page_size) / (1024 * 1024)
            
            # Index usage
            cursor.execute("PRAGMA index_list(orders)")
            stats['orders_indices'] = len(cursor.fetchall())
            
            # WAL mode status
            cursor.execute("PRAGMA journal_mode")
            stats['journal_mode'] = cursor.fetchone()[0]
            
            # Foreign key status
            cursor.execute("PRAGMA foreign_keys")
            stats['foreign_keys_enabled'] = bool(cursor.fetchone()[0])
        
        return stats

class FileRepository:
    """
    File-based repository for trading data
    """
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "klines").mkdir(exist_ok=True)
        (self.base_path / "trades").mkdir(exist_ok=True)
        (self.base_path / "backtests").mkdir(exist_ok=True)
        (self.base_path / "states").mkdir(exist_ok=True)
        (self.base_path / "models").mkdir(exist_ok=True)
    
    def save_klines_parquet(self, klines: List[KlineData], symbol: str, interval: str):
        """Save klines to Parquet file"""
        if not klines:
            return
        
        # Convert to DataFrame
        data = []
        for k in klines:
            data.append({
                'open_time': k.open_time,
                'close_time': k.close_time,
                'open': float(k.open_price),
                'high': float(k.high_price),
                'low': float(k.low_price),
                'close': float(k.close_price),
                'volume': float(k.volume),
                'quote_volume': float(k.quote_asset_volume or 0),
                'trades': k.number_of_trades or 0,
                'taker_buy_base': float(k.taker_buy_base_volume or 0),
                'taker_buy_quote': float(k.taker_buy_quote_volume or 0)
            })
        
        df = pd.DataFrame(data)
        df.set_index('open_time', inplace=True)
        
        # Save to Parquet
        file_path = self.base_path / "klines" / f"{symbol}_{interval}.parquet"
        df.to_parquet(file_path)
        logger.info(f"Saved {len(klines)} klines to {file_path}")
    
    def load_klines_parquet(self, symbol: str, interval: str) -> List[KlineData]:
        """Load klines from Parquet file"""
        file_path = self.base_path / "klines" / f"{symbol}_{interval}.parquet"
        
        if not file_path.exists():
            return []
        
        df = pd.read_parquet(file_path)
        
        klines = []
        for idx, row in df.iterrows():
            kline = KlineData(
                symbol=symbol,
                interval=interval,
                open_time=idx,
                close_time=row['close_time'],
                open_price=Decimal(str(row['open'])),
                high_price=Decimal(str(row['high'])),
                low_price=Decimal(str(row['low'])),
                close_price=Decimal(str(row['close'])),
                volume=Decimal(str(row['volume'])),
                quote_asset_volume=Decimal(str(row['quote_volume'])),
                number_of_trades=int(row['trades']),
                taker_buy_base_volume=Decimal(str(row['taker_buy_base'])),
                taker_buy_quote_volume=Decimal(str(row['taker_buy_quote']))
            )
            klines.append(kline)
        
        return klines
    
    def save_json(self, data: Any, filename: str, subdir: str = ""):
        """Save data as JSON file"""
        path = self.base_path
        if subdir:
            path = path / subdir
            path.mkdir(exist_ok=True)
        
        file_path = path / f"{filename}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, cls=DecimalEncoder, indent=2)
        
        logger.info(f"Saved JSON data to {file_path}")
    
    def load_json(self, filename: str, subdir: str = "") -> Optional[Any]:
        """Load data from JSON file"""
        path = self.base_path
        if subdir:
            path = path / subdir
        
        file_path = path / f"{filename}.json"
        if not file_path.exists():
            return None
        
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def save_pickle(self, data: Any, filename: str, subdir: str = ""):
        """Save data as pickle file"""
        path = self.base_path
        if subdir:
            path = path / subdir
            path.mkdir(exist_ok=True)
        
        file_path = path / f"{filename}.pkl"
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved pickle data to {file_path}")
    
    def load_pickle(self, filename: str, subdir: str = "") -> Optional[Any]:
        """Load data from pickle file"""
        path = self.base_path
        if subdir:
            path = path / subdir
        
        file_path = path / f"{filename}.pkl"
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class HybridRepository:
    """
    Hybrid repository using both SQLite and file storage
    """
    
    def __init__(self, db_path: str = "trading_bot.db", file_path: str = "data"):
        self.db_repo = SQLiteRepository(db_path)
        self.file_repo = FileRepository(file_path)
    
    def save_klines(self, klines: List[KlineData]) -> int:
        """Save klines to both database and Parquet files"""
        if not klines:
            return 0
        
        # Save to database for queries
        db_count = self.db_repo.save_klines(klines)
        
        # Group by symbol and interval for Parquet files
        grouped = {}
        for kline in klines:
            key = (kline.symbol, kline.interval)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(kline)
        
        # Save each group to Parquet
        for (symbol, interval), group_klines in grouped.items():
            self.file_repo.save_klines_parquet(group_klines, symbol, interval)
        
        return db_count
    
    def load_klines(self, symbol: str, interval: str, 
                   prefer_parquet: bool = True, **kwargs) -> List[KlineData]:
        """Load klines with preference for Parquet or database"""
        if prefer_parquet:
            # Try Parquet first
            klines = self.file_repo.load_klines_parquet(symbol, interval)
            if klines:
                return klines
        
        # Fallback to database
        return self.db_repo.load_klines(symbol, interval, **kwargs)
    
    def save_backtest_result(self, result: BacktestResult) -> int:
        """Save backtest result to database and file"""
        # Save to database
        db_id = self.db_repo.save_backtest_result(result)
        
        # Save detailed result to JSON file
        result_data = {
            'strategy_name': result.strategy_name,
            'symbol': result.symbol,
            'start_date': utc_isoformat(result.start_date),
            'end_date': utc_isoformat(result.end_date),
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'metrics': result.metrics.__dict__,
            'parameters': result.parameters,
            'trades': [trade.__dict__ for trade in result.trades],
            'equity_curve': [point.__dict__ for point in result.equity_curve]
        }
        
        filename = f"{result.strategy_name}_{result.symbol}_{result.start_date.strftime('%Y%m%d')}"
        self.file_repo.save_json(result_data, filename, "backtests")
        
        return db_id

# Convenience functions

def get_default_repository() -> HybridRepository:
    """Get default repository instance"""
    return HybridRepository()

def backup_database(db_path: str = "trading_bot.db", backup_path: str = None):
    """Create database backup"""
    if backup_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = f"trading_bot_backup_{timestamp}.db"
    
    import shutil
    shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to {backup_path}")

def cleanup_old_data(repo: HybridRepository, days_to_keep: int = 90):
    """Clean up old data from database using enhanced cleanup method"""
    repo.db_repo.cleanup_old_data(days_to_keep)
