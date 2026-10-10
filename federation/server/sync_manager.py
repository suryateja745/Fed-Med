"""
Concurrency Locks, Thread-Safe Ingestion Queue, and Round Synchronization for FedMed Server.
Prevents race conditions when multiple hospital clients upload parameters simultaneously,
manages round barriers, and handles stale or out-of-sync parameter submissions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np

from federation.utils.logger import setup_logger



class StalePolicy(str, Enum):
    """Policy for handling stale or out-of-order client parameter uploads."""
    REJECT = "reject"
    BUFFER = "buffer"
    DROP = "drop"


@dataclass
class PendingClientUpdate:
    """Container for a single client's uploaded model update and metadata."""
    hospital_id: str
    round_num: int
    parameters: List[np.ndarray]
    num_examples: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RoundSyncManager:
    """
    Thread-safe synchronization manager and ingestion queue for federated learning rounds.

    Responsibilities:
    - Protect parameter ingestion queue with re-entrant locks (RLock).
    - Coordinate round barriers using Condition variables.
    - Validate round sequence and reject/buffer stale parameter uploads.
    - Provide thread-safe telemetry of active uploads and participating hospital nodes.
    """

    def __init__(
        self,
        initial_round: int = 1,
        min_clients_per_round: int = 2,
        stale_policy: Union[StalePolicy, str] = StalePolicy.REJECT,
        max_buffer_size: int = 50,
    ) -> None:
        self.current_round = initial_round
        self.min_clients_per_round = max(1, min_clients_per_round)
        self.stale_policy = StalePolicy(stale_policy)
        self.max_buffer_size = max_buffer_size

        self.logger = setup_logger(name="RoundSyncManager")

        # Concurrency primitives
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)

        # Ingestion queue: hospital_id -> PendingClientUpdate for current round
        self._active_uploads: Dict[str, PendingClientUpdate] = {}

        # Stale & future upload buffers: round_num -> list of PendingClientUpdate
        self._stale_buffer: Dict[int, List[PendingClientUpdate]] = {}
        self._future_buffer: Dict[int, List[PendingClientUpdate]] = {}

        # Historical tracking
        self._completed_rounds: Set[int] = set()
        self._client_latest_round: Dict[str, int] = {}
        self._stale_rejections_count: int = 0
        self._successful_uploads_count: int = 0

    def submit_update(
        self,
        hospital_id: str,
        round_num: int,
        parameters: List[np.ndarray],
        num_examples: int,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """
        Thread-safely submit a client's trained model parameters to the server ingestion queue.

        Args:
            hospital_id: Unique hospital client identifier.
            round_num: Federated round for which the update was trained.
            parameters: List of NumPy parameter arrays.
            num_examples: Number of local training samples.
            metrics: Optional dictionary of local client training/validation metrics.

        Returns:
            Tuple of (success: bool, status_message: str).
        """
        with self._lock:
            metrics_dict = metrics or {}
            update = PendingClientUpdate(
                hospital_id=hospital_id,
                round_num=round_num,
                parameters=[arr.copy() for arr in parameters],
                num_examples=num_examples,
                metrics=metrics_dict,
            )

            # Check 1: Stale parameter upload (from past rounds)
            if round_num < self.current_round or round_num in self._completed_rounds:
                self._stale_rejections_count += 1
                msg = (
                    f"Rejected stale update from '{hospital_id}' for Round {round_num} "
                    f"(Server is currently on Round {self.current_round})."
                )
                self.logger.warning(f"[SyncManager] {msg}")

                if self.stale_policy == StalePolicy.BUFFER:
                    buf = self._stale_buffer.setdefault(round_num, [])
                    if len(buf) < self.max_buffer_size:
                        buf.append(update)

                return False, msg

            # Check 2: Early upload from a future round
            if round_num > self.current_round:
                msg = (
                    f"Buffered future update from '{hospital_id}' for Round {round_num} "
                    f"(Server is currently on Round {self.current_round})."
                )
                self.logger.info(f"[SyncManager] {msg}")
                buf = self._future_buffer.setdefault(round_num, [])
                if len(buf) < self.max_buffer_size:
                    buf.append(update)
                return True, msg

            # Check 3: Current round upload
            is_overwrite = hospital_id in self._active_uploads
            self._active_uploads[hospital_id] = update
            self._client_latest_round[hospital_id] = round_num
            self._successful_uploads_count += 1

            action = "Overwrote duplicate" if is_overwrite else "Ingested"
            msg = (
                f"{action} parameter update from '{hospital_id}' for Round {round_num} "
                f"({len(self._active_uploads)}/{self.min_clients_per_round} required)."
            )
            self.logger.info(f"[SyncManager] {msg}")

            # Notify threads waiting on the round barrier
            if len(self._active_uploads) >= self.min_clients_per_round:
                self._condition.notify_all()

            return True, msg

    def wait_for_round_completion(
        self,
        round_num: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[bool, List[PendingClientUpdate]]:
        """
        Block calling thread until minimum client uploads have arrived for the specified round.

        Args:
            round_num: Target round number (defaults to current_round).
            timeout: Maximum seconds to wait before timeout (None for infinite).

        Returns:
            Tuple of (is_ready: bool, list_of_updates: List[PendingClientUpdate]).
        """
        target_round = round_num if round_num is not None else self.current_round

        with self._condition:
            if target_round != self.current_round:
                # Return immediately if round already completed
                if target_round in self._completed_rounds:
                    return True, []
                return False, []

            def condition_met() -> bool:
                return len(self._active_uploads) >= self.min_clients_per_round

            ready = self._condition.wait_for(condition_met, timeout=timeout)
            updates = list(self._active_uploads.values()) if ready else []
            return ready, updates

    def get_ready_updates(self, round_num: Optional[int] = None) -> List[PendingClientUpdate]:
        """
        Return a thread-safe copy of all currently ingested updates for the target round.
        """
        with self._lock:
            target_round = round_num if round_num is not None else self.current_round
            if target_round == self.current_round:
                return list(self._active_uploads.values())
            return self._stale_buffer.get(target_round, [])

    def advance_round(self, new_round: Optional[int] = None) -> int:
        """
        Thread-safely close the current round, increment round counter, and ingest buffered updates.

        Args:
            new_round: Optional explicit target round number (defaults to current_round + 1).

        Returns:
            The new active round number.
        """
        with self._lock:
            prev_round = self.current_round
            self._completed_rounds.add(prev_round)

            self.current_round = new_round if new_round is not None else (prev_round + 1)
            self._active_uploads.clear()

            # Process future buffered updates matching new round
            if self.current_round in self._future_buffer:
                buffered = self._future_buffer.pop(self.current_round)
                for update in buffered:
                    self._active_uploads[update.hospital_id] = update
                self.logger.info(
                    f"[SyncManager] Advanced to Round {self.current_round}. "
                    f"Restored {len(buffered)} buffered updates."
                )
            else:
                self.logger.info(f"[SyncManager] Advanced from Round {prev_round} to Round {self.current_round}.")

            self._condition.notify_all()
            return self.current_round

    def get_sync_status(self) -> Dict[str, Any]:
        """
        Return comprehensive concurrency status and telemetry dictionary.
        """
        with self._lock:
            return {
                "current_round": self.current_round,
                "min_clients_required": self.min_clients_per_round,
                "active_uploads_count": len(self._active_uploads),
                "active_hospitals": list(self._active_uploads.keys()),
                "is_round_ready": len(self._active_uploads) >= self.min_clients_per_round,
                "completed_rounds": sorted(list(self._completed_rounds)),
                "stale_rejections": self._stale_rejections_count,
                "successful_uploads": self._successful_uploads_count,
                "future_buffered_rounds": list(self._future_buffer.keys()),
            }

    def reset(self, initial_round: int = 1) -> None:
        """Reset synchronization manager state."""
        with self._lock:
            self.current_round = initial_round
            self._active_uploads.clear()
            self._stale_buffer.clear()
            self._future_buffer.clear()
            self._completed_rounds.clear()
            self._client_latest_round.clear()
            self._stale_rejections_count = 0
            self._successful_uploads_count = 0
            self._condition.notify_all()
