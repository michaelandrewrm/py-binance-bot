"""
Console icons for the trading bot interface.

This module defines reusable emoji icons that can be used throughout the application
for consistent visual representation in console output, logs, and user interfaces.

Usage:
    from constants.icons import Icon

    # Clean usage without .value
    print(f"{Icon.ROBOT} Trading bot started")

    # With rich console
    console.print(f"{Icon.SUCCESS} Operation completed successfully")

    # In logger messages
    logger.info(f"{Icon.WARNING} Risk limit approaching")

    # Direct string conversion
    status = str(Icon.RUNNING)  # Returns "🟢"

    # Backward compatibility (.value still works)
    print(f"{Icon.BRAIN.value} AI analysis")
"""

from enum import Enum


class Icon(Enum):
    """
    Emoji icons for console output and user interface elements.

    Each enum member maps to a corresponding emoji that represents
    specific concepts, states, or actions in the trading bot system.
    """

    # === Core Trading & Finance ===
    ROBOT = "🤖"
    MONEY_BAG = "💰"
    MONEY_FLYING = "💸"
    TRENDING_UP = "📈"
    TRENDING_DOWN = "📉"
    CHART = "💹"
    BAR_CHART = "📊"
    CREDIT_CARD = "💳"
    COIN = "🪙"
    BALANCE = "⚖️"
    TARGET = "🎯"

    # === Grid Trading ===
    CHAIN = "🔗"
    LIGHTNING = "⚡"
    RULER = "📏"
    ABACUS = "🧮"
    GEAR = "⚙️"
    CYCLE = "🔄"

    # === AI & Machine Learning ===
    BRAIN = "🧠"
    CRYSTAL_BALL = "🔮"
    DICE = "🎲"
    SATELLITE = "📡"
    MICROSCOPE = "🔬"
    RULER_TRIANGLE = "📐"
    DNA = "🧬"

    # === Safety & Risk Management ===
    SHIELD = "🛡️"
    ALARM = "🚨"
    STOP_SIGN = "🛑"
    WARNING = "⚠️"
    DOOR = "🚪"
    LOCK = "🔒"
    KEY = "🔐"
    SAVE = "💾"

    # === Technical & Development ===
    COMPUTER = "🖥️"
    MOBILE = "📱"
    GLOBE = "🌐"
    ANTENNA = "📡"
    WRENCH = "🔧"
    TOOLS = "🛠️"
    SNAKE = "🐍"
    MEMO = "📝"
    CLIPBOARD = "📋"

    # === Status & State Indicators ===
    SUCCESS = "✅"
    ERROR = "❌"
    PAUSE = "⏸️"
    PLAY = "▶️"
    STOP = "⏹️"
    SEARCH = "🔍"
    EYE = "👁️"
    WATCH = "📡"

    # === Time & Scheduling ===
    ALARM_CLOCK = "⏰"
    STOPWATCH = "⏱️"
    HOURGLASS = "⏳"
    CLOCK = "🕐"
    CALENDAR = "📅"

    # === Performance & Results ===
    TROPHY = "🏆"
    METRICS = "📊"
    GROWTH = "📈"
    TEST_TUBE = "🧪"
    RESEARCH = "🔬"

    # === Communication & Alerts ===
    MEGAPHONE = "📢"
    BELL = "🔔"
    ENVELOPE = "📨"
    OUTBOX = "📤"
    INBOX = "📥"
    CHAT = "💬"

    # === Energy & Activity ===
    BATTERY = "🔋"
    LIGHTNING_BOLT = "⚡"
    FIRE = "🔥"
    SNOWFLAKE = "❄️"
    THERMOMETER = "🌡️"

    # === Navigation & Control ===
    CONTROLLER = "🎮"
    JOYSTICK = "🕹️"
    RADIO_BUTTON = "🔘"
    WHITE_CIRCLE = "⚪"
    GREEN_CIRCLE = "🟢"
    RED_CIRCLE = "🔴"
    YELLOW_CIRCLE = "🟡"

    # === Special Purpose ===
    SPARKLES = "✨"
    ROCKET = "🚀"
    CONSTRUCTION = "🚧"
    PACKAGE = "📦"
    FOLDER = "📁"
    FILE = "📄"
    LINK = "🔗"
    MAGIC_WAND = "🪄"

    def __str__(self) -> str:
        """Return the emoji value as string for easy printing."""
        return self.value

    def __repr__(self) -> str:
        """Return a detailed representation of the icon."""
        return f"Icon.{self.name}('{self.value}')"

    def __format__(self, format_spec: str) -> str:
        """Support for f-string formatting without .value"""
        return format(self.value, format_spec)

    def __add__(self, other: str) -> str:
        """Support string concatenation: Icon.ROBOT + ' text'"""
        return self.value + str(other)

    def __radd__(self, other: str) -> str:
        """Support reverse string concatenation: 'text ' + Icon.ROBOT"""
        return str(other) + self.value


# Convenience aliases for commonly used icons
class StatusIcon:
    """Quick access to status-related icons."""

    RUNNING = Icon.GREEN_CIRCLE
    STOPPED = Icon.RED_CIRCLE
    PAUSED = Icon.YELLOW_CIRCLE
    SUCCESS = Icon.SUCCESS
    ERROR = Icon.ERROR
    WARNING = Icon.WARNING


class TradingIcon:
    """Quick access to trading-related icons."""

    BUY = Icon.TRENDING_UP
    SELL = Icon.TRENDING_DOWN
    PROFIT = Icon.MONEY_BAG
    LOSS = Icon.MONEY_FLYING
    GRID = Icon.CHAIN
    AI = Icon.BRAIN
    ROBOT = Icon.ROBOT


class SafetyIcon:
    """Quick access to safety-related icons."""

    SHIELD = Icon.SHIELD
    ALARM = Icon.ALARM
    STOP = Icon.STOP_SIGN
    WARNING = Icon.WARNING
    EMERGENCY = Icon.FIRE
