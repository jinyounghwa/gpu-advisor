"""
성능 모니터링: 실행 시간, 메모리, 데이터 통계
"""

import time
import json
import logging
import resource
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """성능 메트릭"""
    component: str
    start_time: str
    end_time: str
    elapsed_sec: float
    memory_mb: float
    peak_memory_mb: float
    success: bool
    data_count: int = 0
    error_msg: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceMonitor:
    """실행 성능 모니터링"""

    def __init__(self, log_dir: Path = Path("data/gpu-advisor/logs/performance")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list[PerformanceMetrics] = []

    def _get_memory_mb(self) -> float:
        """현재 메모리 사용량 (MB)"""
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # macOS: ru_maxrss는 bytes, Linux: KB
        maxrss = usage.ru_maxrss
        if sys.platform == "darwin":
            return maxrss / 1024.0 / 1024.0
        return maxrss / 1024.0

    def record(
        self,
        component: str,
        elapsed_sec: float,
        success: bool,
        data_count: int = 0,
        error_msg: str | None = None,
    ) -> PerformanceMetrics:
        """성능 메트릭 기록"""
        memory_mb = self._get_memory_mb()

        metric = PerformanceMetrics(
            component=component,
            start_time=datetime.now().isoformat(),
            end_time=datetime.now().isoformat(),
            elapsed_sec=elapsed_sec,
            memory_mb=memory_mb,
            peak_memory_mb=memory_mb,
            success=success,
            data_count=data_count,
            error_msg=error_msg,
        )

        self.metrics.append(metric)
        logger.info(
            f"📊 {component}: {elapsed_sec:.2f}s, 메모리 {memory_mb:.1f}MB, "
            f"데이터 {data_count}개, {'✓' if success else '✗'}"
        )

        return metric

    def save_daily_report(self) -> str:
        """일일 성능 리포트 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        report_file = self.log_dir / f"performance-{today}.json"

        report = {
            "date": today,
            "timestamp": datetime.now().isoformat(),
            "total_components": len(self.metrics),
            "successful": sum(1 for m in self.metrics if m.success),
            "failed": sum(1 for m in self.metrics if not m.success),
            "total_elapsed_sec": sum(m.elapsed_sec for m in self.metrics),
            "peak_memory_mb": max((m.memory_mb for m in self.metrics), default=0),
            "total_data_items": sum(m.data_count for m in self.metrics),
            "metrics": [m.to_dict() for m in self.metrics],
        }

        report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        logger.info(f"✓ 성능 리포트 저장: {report_file}")

        return str(report_file)

    def summary(self) -> dict[str, Any]:
        """성능 요약"""
        if not self.metrics:
            return {}

        total_time = sum(m.elapsed_sec for m in self.metrics)
        successful = sum(1 for m in self.metrics if m.success)

        return {
            "total_components": len(self.metrics),
            "success_rate": f"{successful / len(self.metrics):.1%}",
            "total_time_sec": f"{total_time:.2f}",
            "avg_time_per_component_sec": f"{total_time / len(self.metrics):.2f}",
            "peak_memory_mb": f"{max(m.memory_mb for m in self.metrics):.1f}",
            "total_items_collected": sum(m.data_count for m in self.metrics),
        }


# 전역 모니터 인스턴스 (run_daily.py에서 사용)
_global_monitor: PerformanceMonitor | None = None


def get_monitor() -> PerformanceMonitor:
    """전역 모니터 인스턴스 가져오기"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def reset_monitor():
    """모니터 초기화"""
    global _global_monitor
    _global_monitor = PerformanceMonitor()
