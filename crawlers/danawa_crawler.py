"""
다나와 GPU 가격 크롤러
매일 GPU 가격 정보 수집
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DanawaCrawler:
    """다나와 GPU 가격 크롤러"""

    def __init__(self, output_dir: str = "data/raw/danawa"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 수집 대상 GPU 모델
        self.target_gpus = [
            "RTX 5090",
            "RTX 5080",
            "RTX 5070 Ti",
            "RTX 5070",
            "RTX 5060 Ti",
            "RTX 5060",
            "RTX 5050",
            "RTX 4090",
            "RTX 4080",
            "RTX 4070 Ti",
            "RTX 4070",
            "RTX 4060 Ti",
            "RTX 4060",
            "RX 9070 XT",
            "RX 9060 XT",
            "RX 7900 XTX",
            "RX 7900 XT",
            "RX 7800 XT",
            "RX 7700 XT",
            "RX 7600",
            "RX 6600",
            "Arc B580",
            "Arc B570",
            "Arc A770",
        ]

        # User-Agent
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

    # 다나와 GPU 카테고리 코드
    DANAWA_CATE = "112753"

    def search_gpu_price(self, gpu_model: str) -> dict:
        """
        특정 GPU 모델의 최저가 검색 (다나와 실시간 크롤링)

        Args:
            gpu_model: GPU 모델명 (예: RTX 5060)

        Returns:
            가격 정보 딕셔너리
        """
        try:
            result = self._crawl_danawa(gpu_model)
            if result:
                logger.info(f"✓ {gpu_model}: {result['lowest_price']:,}원 ({result['product_name']})")
                return result
            logger.warning(f"⚠ {gpu_model}: 크롤링 실패, 건너뜀")
            return None

        except Exception as e:
            logger.error(f"✗ {gpu_model} 크롤링 오류: {e}")
            return None

    def _crawl_danawa(self, gpu_model: str) -> dict:
        """다나와 검색 페이지에서 GPU 최저가 파싱"""
        search_url = "https://search.danawa.com/dsearch.php"
        params = {
            "query": gpu_model,
            "tab": "goods",
            "page": 1,
            "limit": 40,
        }
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.danawa.com/",
            "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

        response = requests.get(search_url, params=params, headers=headers, timeout=15)
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
                f"https://prod.danawa.com/info/?pcode={pcode}&cate={self.DANAWA_CATE}"
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
        """모든 GPU 모델 크롤링"""
        logger.info("=" * 80)
        logger.info(f"다나와 GPU 가격 크롤링 시작 - {datetime.now()}")
        logger.info("=" * 80)

        results = []

        for gpu_model in self.target_gpus:
            product_info = self.search_gpu_price(gpu_model)
            if product_info:
                results.append(product_info)
            time.sleep(1)  # 서버 부하 방지

        logger.info(f"\n✓ 총 {len(results)}개 제품 수집 완료")
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
    crawler = DanawaCrawler()
    products = crawler.crawl_all()
    crawler.save(products)
