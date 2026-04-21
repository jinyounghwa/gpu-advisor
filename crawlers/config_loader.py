"""
크롤러 설정 로더
JSON 기반 중앙 설정 관리
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ConfigLoader:
    """설정 파일 로더"""

    def __init__(self, config_file: Path = None):
        if config_file is None:
            # 프로젝트 루트 기준 기본 경로
            config_file = Path(__file__).parent.parent / "config" / "crawler_config.json"

        self.config_file = Path(config_file)
        self.config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        """설정 파일 로드"""
        if not self.config_file.exists():
            logger.warning(f"설정 파일 없음: {self.config_file} (기본 설정 사용)")
            return {}

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            logger.info(f"✓ 설정 로드: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"설정 로드 실패: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """설정값 조회"""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default

    def get_danawa_config(self) -> dict:
        """다나와 크롤러 설정"""
        return self.get("danawa", {})

    def get_retry_config(self) -> dict:
        """재시도 설정"""
        return self.get("danawa.retry", {})

    def get_cache_config(self) -> dict:
        """캐시 설정"""
        return self.get("cache", {})

    def get_performance_config(self) -> dict:
        """성능 모니터링 설정"""
        return self.get("performance", {})


# 전역 설정 로더 인스턴스
_global_loader: ConfigLoader | None = None


def get_loader() -> ConfigLoader:
    """전역 설정 로더 인스턴스 가져오기"""
    global _global_loader
    if _global_loader is None:
        _global_loader = ConfigLoader()
    return _global_loader


def reload_config(config_file: Path | None = None):
    """설정 다시 로드"""
    global _global_loader
    _global_loader = ConfigLoader(config_file)
    return _global_loader
