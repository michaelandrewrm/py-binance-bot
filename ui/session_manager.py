"""
Session management for trading bot CLI

This module provides thread-safe session management with automatic cleanup
and proper lifecycle management for trading sessions.
"""

import threading
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

from core.grid_engine import GridConfig
from .output import session_output

# Session management operates mainly as debug/internal logging


class TradingMode(Enum):
    MANUAL = "manual"
    AI = "ai"


# Trading Session Management Classes
class TradingSessionStatus(Enum):
    """Trading session status"""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class GridParameters:
    """Grid trading parameters"""

    symbol: str
    lower_bound: Decimal
    upper_bound: Decimal
    grid_count: int
    investment_per_grid: Decimal
    center_price: Optional[Decimal] = None
    mode: TradingMode = TradingMode.MANUAL
    confidence: Optional[float] = None  # For AI suggestions

    def to_grid_config(self) -> GridConfig:
        """Convert to GridConfig"""
        center = self.center_price or (self.lower_bound + self.upper_bound) / 2
        spacing = (self.upper_bound - self.lower_bound) / self.grid_count

        return GridConfig(
            center_price=center,
            grid_spacing=spacing,
            num_levels_up=self.grid_count // 2,
            num_levels_down=self.grid_count // 2,
            order_amount=self.investment_per_grid,
            max_position_size=self.investment_per_grid * self.grid_count,
        )


@dataclass
class TradingSession:
    """Runtime trading session management"""

    session_id: str
    symbol: str
    parameters: GridParameters
    status: TradingSessionStatus = TradingSessionStatus.STOPPED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_limit: Optional[timedelta] = None
    pause_time: Optional[datetime] = None
    total_paused_duration: timedelta = timedelta()

    # Trading metrics
    trades_count: int = 0
    profit_loss: Decimal = Decimal("0")
    current_positions: int = 0

    def is_active(self) -> bool:
        """Check if session is actively trading"""
        return self.status in [
            TradingSessionStatus.RUNNING,
            TradingSessionStatus.STARTING,
        ]

    def is_time_expired(self) -> bool:
        """Check if duration limit is exceeded"""
        if not self.duration_limit or not self.start_time:
            return False

        elapsed = (
            datetime.now(timezone.utc) - self.start_time - self.total_paused_duration
        )
        return elapsed >= self.duration_limit

    def get_elapsed_time(self) -> timedelta:
        """Get elapsed trading time (excluding paused time)"""
        if not self.start_time:
            return timedelta()

        end = self.end_time or datetime.now(timezone.utc)

        if self.status == TradingSessionStatus.PAUSED and self.pause_time:
            end = self.pause_time

        return end - self.start_time - self.total_paused_duration


@dataclass
class SessionStats:
    """Statistics for session management"""

    total_created: int = 0
    total_completed: int = 0
    total_errors: int = 0
    active_count: int = 0
    cleanup_count: int = 0


class ThreadSafeSessionManager:
    """Thread-safe session manager with proper cleanup and monitoring"""

    def __init__(self, max_sessions: int = 100, cleanup_threshold: int = 80):
        self.sessions: Dict[str, TradingSession] = {}
        self.active_session: Optional[str] = None
        self._lock = threading.RLock()  # Use RLock for nested locking
        self.max_sessions = max_sessions
        self.cleanup_threshold = cleanup_threshold
        self.stats = SessionStats()

        # Start cleanup thread
        self._cleanup_thread = threading.Thread(
            target=self._periodic_cleanup, daemon=True, name="session-cleanup"
        )
        self._stop_cleanup = threading.Event()
        self._cleanup_thread.start()

    @contextmanager
    def _session_lock(self):
        """Context manager for thread-safe operations"""
        acquired = self._lock.acquire(timeout=5.0)  # Timeout to prevent deadlocks
        if not acquired:
            raise RuntimeError("Failed to acquire session lock")
        try:
            yield
        finally:
            self._lock.release()

    def create_session(
        self,
        symbol: str,
        parameters: GridParameters,
        duration_limit: Optional[timedelta] = None,
    ) -> str:
        """Create a new trading session with automatic cleanup"""
        session_id = (
            f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:-3]}"
        )

        with self._session_lock():
            # Clean up old sessions if we're approaching the limit
            if len(self.sessions) >= self.cleanup_threshold:
                self._cleanup_old_sessions()

            # Check if we're still at max capacity
            if len(self.sessions) >= self.max_sessions:
                raise RuntimeError(
                    f"Maximum session limit reached ({self.max_sessions})"
                )

            session = TradingSession(
                session_id=session_id,
                symbol=symbol,
                parameters=parameters,
                duration_limit=duration_limit,
            )

            self.sessions[session_id] = session
            self.stats.total_created += 1
            self.stats.active_count = len(
                [s for s in self.sessions.values() if s.is_active()]
            )

        session_output.debug_only(f"Created session {session_id} for {symbol}")
        return session_id

    def _cleanup_old_sessions(self) -> int:
        """Remove old inactive sessions to prevent memory leaks"""
        if len(self.sessions) < self.cleanup_threshold:
            return 0

        # Find sessions to clean up (inactive and old)
        now = datetime.now(timezone.utc)
        sessions_to_remove = []

        for session_id, session in self.sessions.items():
            if session.is_active():
                continue  # Don't remove active sessions

            # Remove sessions that are old or in terminal states
            should_remove = False

            if session.status in [
                TradingSessionStatus.COMPLETED,
                TradingSessionStatus.ERROR,
            ]:
                should_remove = True
            elif session.end_time and (now - session.end_time) > timedelta(hours=24):
                should_remove = True  # Remove sessions older than 24 hours
            elif not session.start_time and session_id != self.active_session:
                # Remove old unused sessions (created but never started)
                creation_time = self._extract_creation_time(session_id)
                if creation_time and (now - creation_time) > timedelta(hours=1):
                    should_remove = True

            if should_remove:
                sessions_to_remove.append(session_id)

        # Sort by age and remove oldest first
        sessions_to_remove.sort(key=lambda sid: self._extract_creation_time(sid) or now)

        # Remove up to 20% of sessions or until we're under threshold
        max_to_remove = min(
            len(sessions_to_remove),
            max(1, len(self.sessions) - self.cleanup_threshold + 10),
        )

        removed_count = 0
        for session_id in sessions_to_remove[:max_to_remove]:
            # Double-check session is not active before removing
            session = self.sessions.get(session_id)
            if session and not session.is_active():
                del self.sessions[session_id]
                removed_count += 1

                # Clear active session if it was removed
                if self.active_session == session_id:
                    self.active_session = None

        if removed_count > 0:
            self.stats.cleanup_count += removed_count
            self.stats.active_count = len(
                [s for s in self.sessions.values() if s.is_active()]
            )
            session_output.debug_only(f"Cleaned up {removed_count} old sessions")

        return removed_count

    def _extract_creation_time(self, session_id: str) -> Optional[datetime]:
        """Extract creation time from session ID"""
        try:
            # Session ID format: SYMBOL_YYYYMMDD_HHMMSS_mmm
            parts = session_id.split("_")
            if len(parts) >= 4:
                date_str = parts[1]  # YYYYMMDD
                time_str = parts[2]  # HHMMSS
                ms_str = parts[3] if len(parts) > 3 else "000"  # mmm

                datetime_str = f"{date_str}_{time_str}_{ms_str}"
                return datetime.strptime(datetime_str, "%Y%m%d_%H%M%S_%f").replace(
                    tzinfo=timezone.utc
                )
        except (ValueError, IndexError):
            pass
        return None

    def _periodic_cleanup(self):
        """Periodic cleanup task running in background thread"""
        while not self._stop_cleanup.wait(300):  # Run every 5 minutes
            try:
                with self._session_lock():
                    self._cleanup_old_sessions()
            except Exception as e:
                session_output.error(f"Error in periodic cleanup: {e}")

    def get_session(self, session_id: str) -> Optional[TradingSession]:
        """Thread-safe session retrieval"""
        with self._session_lock():
            return self.sessions.get(session_id)

    def get_active_session(self) -> Optional[TradingSession]:
        """Get currently active session"""
        with self._session_lock():
            if self.active_session:
                return self.sessions.get(self.active_session)
            return None

    def set_active_session(self, session_id: str) -> bool:
        """Thread-safe active session setting with validation"""
        with self._session_lock():
            session = self.sessions.get(session_id)
            if not session:
                return False

            # Warn if setting a non-active session as active
            if (
                not session.is_active()
                and session.status != TradingSessionStatus.STOPPED
            ):
                session_output.warning(
                    f"Setting inactive session as active: {session_id} (status: {session.status})"
                )

            # Deactivate previous session if different
            if self.active_session and self.active_session != session_id:
                old_session = self.sessions.get(self.active_session)
                if old_session and old_session.is_active():
                    session_output.debug_only(
                        f"Deactivating previous session: {self.active_session}"
                    )

            self.active_session = session_id
            return True

    def list_sessions(
        self,
        include_inactive: bool = False,
        status_filter: Optional[TradingSessionStatus] = None,
    ) -> List[TradingSession]:
        """List sessions with optional filtering"""
        with self._session_lock():
            sessions = list(self.sessions.values())

            if not include_inactive:
                sessions = [s for s in sessions if s.is_active()]

            if status_filter:
                sessions = [s for s in sessions if s.status == status_filter]

            # Sort by creation time (newest first)
            sessions.sort(
                key=lambda s: self._extract_creation_time(s.session_id)
                or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )

            return sessions

    def get_session_count(self) -> Dict[str, int]:
        """Get session counts by status"""
        with self._session_lock():
            counts = {
                "total": len(self.sessions),
                "active": 0,
                "stopped": 0,
                "running": 0,
                "paused": 0,
                "error": 0,
            }

            for session in self.sessions.values():
                if session.is_active():
                    counts["active"] += 1

                status_name = session.status.value.lower()
                if status_name in counts:
                    counts[status_name] += 1

            return counts

    def update_session_status(
        self, session_id: str, status: TradingSessionStatus
    ) -> bool:
        """Update session status with validation"""
        with self._session_lock():
            session = self.sessions.get(session_id)
            if not session:
                return False

            old_status = session.status
            session.status = status

            # Update timestamps based on status changes
            now = datetime.now(timezone.utc)

            if (
                status == TradingSessionStatus.RUNNING
                and old_status != TradingSessionStatus.RUNNING
            ):
                if not session.start_time:
                    session.start_time = now
                elif old_status == TradingSessionStatus.PAUSED:
                    # Resume from pause - add paused duration
                    if session.pause_time:
                        session.total_paused_duration += now - session.pause_time
                        session.pause_time = None

            elif status == TradingSessionStatus.PAUSED:
                session.pause_time = now

            elif status in [
                TradingSessionStatus.STOPPED,
                TradingSessionStatus.COMPLETED,
                TradingSessionStatus.ERROR,
            ]:
                if not session.end_time:
                    session.end_time = now
                if session.pause_time:
                    session.total_paused_duration += now - session.pause_time
                    session.pause_time = None

                # Update stats
                if status == TradingSessionStatus.COMPLETED:
                    self.stats.total_completed += 1
                elif status == TradingSessionStatus.ERROR:
                    self.stats.total_errors += 1

            # Update active count
            self.stats.active_count = len(
                [s for s in self.sessions.values() if s.is_active()]
            )

            session_output.debug_only(
                f"Session {session_id} status: {old_status.value} -> {status.value}"
            )
            return True

    def force_cleanup(self) -> int:
        """Force immediate cleanup of old sessions"""
        with self._session_lock():
            return self._cleanup_old_sessions()

    def shutdown(self):
        """Shutdown session manager and cleanup resources"""
        session_output.debug_only("Shutting down session manager")
        self._stop_cleanup.set()

        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5.0)

        with self._session_lock():
            active_sessions = [s for s in self.sessions.values() if s.is_active()]
            if active_sessions:
                session_output.warning(
                    f"Shutting down with {len(active_sessions)} active sessions"
                )

    def get_stats(self) -> SessionStats:
        """Get session management statistics"""
        with self._session_lock():
            # Update current active count
            self.stats.active_count = len(
                [s for s in self.sessions.values() if s.is_active()]
            )
            return self.stats
