"""
HTTP 캐싱 및 로컬 캐시 관리
- 캐시 유효 기간 설정
- 조건부 요청 (ETag, Last-Modified)
- 로컬 파일 캐시
"""

import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import requests

logger = logging.getLogger(__name__)


class HTTPCacheManager:
    """HTTP 응답 캐싱 관리"""

    def __init__(self, cache_dir: Path = Path("data/cache/http"), ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.metadata_file = self.cache_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """캐시 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text())
            except Exception as e:
                logger.warning(f"캐시 메타데이터 로드 실패: {e}")
        return {}

    def _save_metadata(self):
        """캐시 메타데이터 저장"""
        try:
            self.metadata_file.write_text(json.dumps(self.metadata, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.warning(f"캐시 메타데이터 저장 실패: {e}")

    def _get_cache_key(self, url: str, params: dict | None = None) -> str:
        """캐시 키 생성"""
        cache_str = url
        if params:
            cache_str += json.dumps(params, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def is_cache_valid(self, cache_key: str) -> bool:
        """캐시 유효성 확인"""
        if cache_key not in self.metadata:
            return False

        cached_at = datetime.fromisoformat(self.metadata[cache_key]["cached_at"])
        return datetime.now() < cached_at + self.ttl

    def get_cached(self, cache_key: str) -> Optional[str]:
        """캐시에서 내용 가져오기"""
        if not self.is_cache_valid(cache_key):
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                data = json.loads(cache_file.read_text())
                logger.info(f"📦 캐시 히트: {cache_key[:8]}...")
                return data["content"]
            except Exception as e:
                logger.warning(f"캐시 읽기 실패: {e}")
        return None

    def set_cached(self, cache_key: str, content: str, metadata: dict | None = None):
        """캐시에 내용 저장"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        cache_data = {
            "content": content,
            "metadata": metadata or {},
        }

        try:
            cache_file.write_text(json.dumps(cache_data, ensure_ascii=False))
            self.metadata[cache_key] = {
                "cached_at": datetime.now().isoformat(),
                "size_bytes": len(content.encode()),
                "metadata": metadata or {},
            }
            self._save_metadata()
            logger.info(f"💾 캐시 저장: {cache_key[:8]}...")
        except Exception as e:
            logger.warning(f"캐시 저장 실패: {e}")

    def get_with_fallback(
        self,
        url: str,
        headers: dict | None = None,
        params: dict | None = None,
        timeout: int = 15,
    ) -> Optional[requests.Response]:
        """
        캐시를 우선으로 하는 HTTP 요청

        1. 유효한 캐시가 있으면 반환
        2. 없으면 HTTP 요청
        3. 요청 실패시 만료된 캐시라도 반환 (폴백)

        Returns:
            requests.Response 또는 None
        """
        cache_key = self._get_cache_key(url, params)

        # 캐시 확인
        cached_content = self.get_cached(cache_key)
        if cached_content:
            # 캐시된 텍스트를 Response처럼 변환
            response = requests.Response()
            response._content = cached_content.encode()
            response.status_code = 200
            response.headers["X-Cache"] = "HIT"
            return response

        # HTTP 요청
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()

            # 캐시에 저장
            self.set_cached(
                cache_key,
                response.text,
                metadata={
                    "status_code": response.status_code,
                    "content_type": response.headers.get("content-type"),
                },
            )
            response.headers["X-Cache"] = "MISS"
            return response

        except requests.RequestException as e:
            logger.warning(f"요청 실패: {e} (캐시 폴백 시도...)")

            # 만료된 캐시라도 반환 (폴백)
            cache_file = self.cache_dir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    response = requests.Response()
                    response._content = data["content"].encode()
                    response.status_code = 200
                    response.headers["X-Cache"] = "STALE"
                    logger.info(f"📦 폴백 캐시 사용 (만료됨): {cache_key[:8]}...")
                    return response
                except Exception:
                    pass

            return None

    def clear_expired(self):
        """만료된 캐시 삭제"""
        to_delete = []

        for cache_key, meta in self.metadata.items():
            cached_at = datetime.fromisoformat(meta["cached_at"])
            if datetime.now() >= cached_at + self.ttl:
                to_delete.append(cache_key)
                cache_file = self.cache_dir / f"{cache_key}.json"
                if cache_file.exists():
                    cache_file.unlink()

        for key in to_delete:
            del self.metadata[key]

        if to_delete:
            self._save_metadata()
            logger.info(f"🗑️  {len(to_delete)}개 만료된 캐시 삭제")

    def stats(self) -> dict:
        """캐시 통계"""
        total_size = sum(
            (self.cache_dir / f"{key}.json").stat().st_size
            for key in self.metadata
            if (self.cache_dir / f"{key}.json").exists()
        )

        valid_count = sum(1 for key in self.metadata if self.is_cache_valid(key))

        return {
            "total_cached": len(self.metadata),
            "valid_cached": valid_count,
            "expired": len(self.metadata) - valid_count,
            "total_size_mb": total_size / 1024 / 1024,
        }
