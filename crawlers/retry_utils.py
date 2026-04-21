"""
재시도 로직 및 타임아웃 관리 유틸리티
- exponential backoff 재시도
- 타임아웃 설정
- 에러 추적
"""

import time
import logging
from typing import TypeVar, Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryConfig:
    """재시도 설정"""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_factor: float = 2.0,
        jitter: bool = True,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """지수 백오프 지연 계산"""
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            import random
            delay = delay * (0.5 + random.random())

        return delay


def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig | None = None,
    *args,
    **kwargs,
) -> T:
    """
    지수 백오프 재시도를 통한 함수 실행

    Args:
        func: 실행할 함수
        config: 재시도 설정
        *args, **kwargs: func에 전달할 인자

    Returns:
        함수 반환값

    Raises:
        마지막 예외를 재발생
    """
    if config is None:
        config = RetryConfig()

    last_exception = None

    for attempt in range(config.max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e

            if attempt == config.max_retries:
                logger.error(
                    f"재시도 실패 ({config.max_retries + 1}회 시도): {e}",
                    exc_info=True
                )
                raise

            delay = config.get_delay(attempt)
            logger.warning(
                f"시도 {attempt + 1}/{config.max_retries + 1} 실패: {e} "
                f"({delay:.1f}초 후 재시도)"
            )
            time.sleep(delay)

    raise last_exception or RuntimeError("Unknown retry failure")


class RetryStats:
    """재시도 통계"""

    def __init__(self):
        self.total_attempts = 0
        self.successful_attempts = 0
        self.failed_attempts = 0
        self.total_delay = 0.0
        self.errors: list[dict[str, Any]] = []

    def record_attempt(self, success: bool, delay: float = 0.0, error: Exception | None = None):
        """시도 기록"""
        self.total_attempts += 1
        self.total_delay += delay

        if success:
            self.successful_attempts += 1
        else:
            self.failed_attempts += 1
            if error:
                self.errors.append({
                    "timestamp": datetime.now().isoformat(),
                    "error_type": type(error).__name__,
                    "error_msg": str(error),
                })

    def success_rate(self) -> float:
        """성공률"""
        if self.total_attempts == 0:
            return 0.0
        return self.successful_attempts / self.total_attempts

    def summary(self) -> dict[str, Any]:
        """통계 요약"""
        return {
            "total_attempts": self.total_attempts,
            "successful": self.successful_attempts,
            "failed": self.failed_attempts,
            "success_rate": f"{self.success_rate():.1%}",
            "total_delay_sec": f"{self.total_delay:.2f}",
            "avg_delay_sec": f"{self.total_delay / max(1, self.total_attempts - 1):.2f}",
            "recent_errors": self.errors[-5:] if self.errors else [],
        }
