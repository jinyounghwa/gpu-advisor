import time
from collections import defaultdict
from threading import Lock
from backend.server.schemas import TrainingConfig

class TrainingState:
    def __init__(self):
        self._lock = Lock()
        self.is_training = False
        self.current_step = 0
        self.total_steps = 0
        self.metrics_history = []
        self.start_time = None
        self.config = None

    def reset(self):
        with self._lock:
            self.is_training = False
            self.current_step = 0
            self.total_steps = 0
            self.metrics_history = []
            self.start_time = None
            self.config = None

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "is_training": self.is_training,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "metrics_history": list(self.metrics_history),
                "start_time": self.start_time,
                "config": self.config,
            }

    def configure(self, *, total_steps: int, start_time: float, config: TrainingConfig) -> None:
        with self._lock:
            self.is_training = True
            self.current_step = 0
            self.total_steps = total_steps
            self.start_time = start_time
            self.config = config

    def append_metric(self, metric: dict) -> None:
        with self._lock:
            self.metrics_history.append(metric)
            if "step" in metric:
                self.current_step = metric["step"]

    def stop(self) -> None:
        with self._lock:
            self.is_training = False

    def should_stop(self) -> bool:
        with self._lock:
            return not self.is_training

class RateLimiter:
    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._lock = Lock()
        self._bucket: dict[tuple[str, int], int] = defaultdict(int)

    @staticmethod
    def _current_minute() -> int:
        return int(time.time() // 60)

    def is_allowed(self, client_ip: str) -> tuple[bool, int]:
        minute = self._current_minute()
        with self._lock:
            key = (client_ip, minute)
            self._bucket[key] += 1
            current = self._bucket[key]
            stale = [k for k in self._bucket.keys() if k[1] < minute - 2]
            for k in stale:
                self._bucket.pop(k, None)
        remaining = max(self.per_minute - current, 0)
        return current <= self.per_minute, remaining

training_state = TrainingState()
