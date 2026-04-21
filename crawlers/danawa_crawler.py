"""
다나와 GPU 가격 크롤러
매일 GPU 가격 정보 수집
재시도 로직 및 타임아웃 처리 포함
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import time
import logging
from crawlers.retry_utils import RetryConfig, RetryStats
from crawlers.http_cache import HTTPCacheManager
from crawlers.config_loader import get_loader

logger = logging.getLogger(__name__)


class DanawaCrawler:
    """다나와 GPU 가격 크롤러"""

    def __init__(self, output_dir: str = "data/raw/danawa"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 설정 로더
        config = get_loader()
        danawa_cfg = config.get_danawa_config()
        retry_cfg = config.get_retry_config()
        cache_cfg = config.get_cache_config()

        # 재시도 설정 (config에서 로드)
        self.retry_config = RetryConfig(
            max_retries=retry_cfg.get("max_retries", 3),
            initial_delay=retry_cfg.get("initial_delay_sec", 1.0),
            max_delay=retry_cfg.get("max_delay_sec", 30.0),
            backoff_factor=retry_cfg.get("backoff_factor", 2.0),
            jitter=retry_cfg.get("jitter", True),
        )
        self.retry_stats = RetryStats()

        # HTTP 캐시 (config에서 로드)
        self.http_cache = HTTPCacheManager(
            cache_dir=Path(cache_cfg.get("http_cache_dir", "data/cache/http")),
            ttl_hours=cache_cfg.get("ttl_hours", 24)
        )

        # 수집 대상 GPU 모델 (config에서 로드)
        self.target_gpus = danawa_cfg.get("target_gpus", [])
        if not self.target_gpus:
            logger.warning("설정에서 target_gpus 로드 실패, 기본값 사용")

        # User-Agent (config에서 로드)
        user_agent = danawa_cfg.get("user_agent",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.headers = {
            "User-Agent": user_agent,
            "Referer": "https://www.danawa.com/",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        # 다나와 GPU 카테고리 코드 (config에서 로드)
        self.danawa_cate = danawa_cfg.get("category_code", "112753")
        self.timeout = danawa_cfg.get("timeout_sec", 15)

    def search_gpu_price(self, gpu_model: str) -> dict | None:
        """
        특정 GPU 모델의 최저가 검색 (재시도 로직 포함)

        Args:
            gpu_model: GPU 모델명 (예: RTX 5060)

        Returns:
            가격 정보 딕셔너리
        """
        attempt = 0
        last_error = None

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                result = self._crawl_danawa(gpu_model)
                if result:
                    self.retry_stats.record_attempt(success=True)
                    logger.info(f"✓ {gpu_model}: {result['lowest_price']:,}원 ({result['product_name']})")
                    return result
                logger.warning(f"⚠ {gpu_model}: 데이터 미수집 (시도 {attempt + 1}/{self.retry_config.max_retries + 1})")
                return None

            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                if attempt == self.retry_config.max_retries:
                    self.retry_stats.record_attempt(success=False, error=e)
                    logger.error(f"✗ {gpu_model} 타임아웃 (최종 실패): {e}")
                    return None

                delay = self.retry_config.get_delay(attempt)
                logger.warning(
                    f"⚠ {gpu_model} 타임아웃 (시도 {attempt + 1}/{self.retry_config.max_retries + 1}): {delay:.1f}초 후 재시도"
                )
                time.sleep(delay)

            except Exception as e:
                last_error = e
                self.retry_stats.record_attempt(success=False, error=e)
                logger.error(f"✗ {gpu_model} 크롤링 오류: {e}")
                return None

        return None

    def _crawl_danawa(self, gpu_model: str) -> dict | None:
        """다나와 검색 페이지에서 GPU 최저가 파싱 (캐시 지원)"""
        search_url = "https://search.danawa.com/dsearch.php"
        params = {
            "query": gpu_model,
            "tab": "goods",
            "page": 1,
            "limit": 40,
        }

        # HTTP 캐시 활용
        response = self.http_cache.get_with_fallback(
            search_url,
            headers=self.headers,
            params=params,
            timeout=self.timeout
        )

        if response is None:
            raise requests.RequestException("Failed to fetch and no cache available")

        response.raise_for_status()
        response.encoding = "utf-8"

        soup = BeautifulSoup(response.text, "html.parser")

        # 다나와 검색 결과: li.prod_item
        items = soup.select("li.prod_item")
        if not items:
            return None

        best = None  # 최저가 상품

        for item in items:
            # 상품명 + URL: p.prod_name > a
            name_el = item.select_one("p.prod_name > a")
            if not name_el:
                continue

            product_name = name_el.get_text(strip=True)
            href = name_el.get("href", "")

            # pcode 추출 (예: ?pcode=12345678&keyword=...)
            pcode = None
            for part in href.replace("?", "&").split("&"):
                if part.startswith("pcode="):
                    pcode = part[6:]
                    break
            if not pcode:
                continue

            product_url = (
                f"https://prod.danawa.com/info/?pcode={pcode}&cate={self.danawa_cate}"
            )

            # 가격: p.price_sect > a (텍스트: "6,148,990원")
            price_el = item.select_one("p.price_sect > a")
            if not price_el:
                continue
            price_text = (
                price_el.get_text(strip=True)
                .replace(",", "")
                .replace("원", "")
                .strip()
            )
            if not price_text.isdigit():
                continue
            price = int(price_text)

            # 판매자 수: span.text__number
            seller_el = item.select_one("span.text__number")
            seller_count = 0
            if seller_el:
                s = seller_el.get_text(strip=True)
                seller_count = int(s) if s.isdigit() else 0

            # 재고 상태
            if seller_count == 0:
                stock_status = "out_of_stock"
            elif seller_count <= 3:
                stock_status = "low_stock"
            else:
                stock_status = "in_stock"

            # 제조사: 상품명 첫 단어
            manufacturer = product_name.split()[0] if product_name else "Unknown"

            candidate = {
                "product_name": product_name,
                "manufacturer": manufacturer,
                "chipset": gpu_model,
                "lowest_price": price,
                "seller_count": seller_count,
                "stock_status": stock_status,
                "product_url": product_url,
            }

            # 최저가 상품 선택
            if best is None or price < best["lowest_price"]:
                best = candidate

        return best

    def crawl_all(self) -> list:
        """모든 GPU 모델 크롤링 (캐시 지원)"""
        logger.info("=" * 80)
        logger.info(f"다나와 GPU 가격 크롤링 시작 - {datetime.now()}")
        logger.info("=" * 80)

        results = []
        self.retry_stats = RetryStats()  # 통계 초기화

        for gpu_model in self.target_gpus:
            product_info = self.search_gpu_price(gpu_model)
            if product_info:
                results.append(product_info)
            time.sleep(1)  # 서버 부하 방지

        # 재시도 통계 출력
        retry_stats = self.retry_stats.summary()
        logger.info(f"\n✓ 총 {len(results)}개 제품 수집 완료")
        logger.info(f"재시도 통계: 성공률 {retry_stats['success_rate']} "
                   f"({retry_stats['successful']}/{retry_stats['total_attempts']}) "
                   f"총 지연 {retry_stats['total_delay_sec']}초")

        # 캐시 통계 출력
        cache_stats = self.http_cache.stats()
        logger.info(f"캐시 통계: {cache_stats['valid_cached']}개 유효, "
                   f"{cache_stats['expired']}개 만료, "
                   f"용량 {cache_stats['total_size_mb']:.2f}MB")

        if retry_stats['recent_errors']:
            logger.info(f"최근 에러: {retry_stats['recent_errors'][-1]['error_type']}")

        return results

    def save(self, data: list) -> str:
        """수집 데이터 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.output_dir / f"{today}.json"

        output_data = {
            "date": today,
            "source": "danawa",
            "total_products": len(data),
            "products": data,
            "metadata": {
                "crawled_at": datetime.now().isoformat(),
                "target_models": len(self.target_gpus),
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 저장 완료: {output_file}")
        return str(output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)  # 독립 실행 시에만 설정
    crawler = DanawaCrawler()
    products = crawler.crawl_all()
    crawler.save(products)
