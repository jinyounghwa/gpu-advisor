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

    def search_gpu_price(self, gpu_model: str) -> dict:
        """
        특정 GPU 모델의 최저가 검색

        Args:
            gpu_model: GPU 모델명 (예: RTX 5060)

        Returns:
            가격 정보 딕셔너리
        """
        try:
            # 다나와 검색 API (간소화된 버전)
            # 실제로는 다나와 검색 페이지를 파싱해야 합니다
            search_url = f"http://www.danawa.com/search/?query={gpu_model.replace(' ', '+')}"

            # 실제 크롤링 대신 mock 데이터 반환 (실전에서는 실제 파싱)
            # response = requests.get(search_url, headers=self.headers, timeout=10)
            # soup = BeautifulSoup(response.text, 'html.parser')

            # Mock 데이터 (실제로는 파싱 결과)
            import random

            base_prices = {
                "RTX 5090": 6500000,
                "RTX 5080": 2200000,
                "RTX 5070 Ti": 1600000,
                "RTX 5070": 1100000,
                "RTX 5060 Ti": 750000,
                "RTX 5060": 600000,
                "RTX 5050": 450000,
                "RTX 4090": 2500000,
                "RTX 4080": 1800000,
                "RTX 4070 Ti": 1200000,
                "RTX 4070": 900000,
                "RTX 4060 Ti": 650000,
                "RTX 4060": 500000,
                "RX 9070 XT": 1100000,
                "RX 9060 XT": 700000,
                "RX 7900 XTX": 1400000,
                "RX 7900 XT": 1200000,
                "RX 7800 XT": 900000,
                "RX 7700 XT": 750000,
                "RX 7600": 500000,
                "RX 6600": 350000,
                "Arc B580": 350000,
                "Arc B570": 300000,
                "Arc A770": 450000,
            }

            base_price = base_prices.get(gpu_model, 500000)
            # 가격 변동 추가 (±5%)
            price = int(base_price * (1 + random.uniform(-0.05, 0.05)))

            result = {
                "product_name": f"{gpu_model} 샘플 제품",
                "manufacturer": random.choice(["MSI", "ASUS", "GIGABYTE", "ZOTAC", "PALIT"]),
                "chipset": gpu_model,
                "lowest_price": price,
                "seller_count": random.randint(5, 30),
                "stock_status": random.choice(["in_stock", "low_stock", "out_of_stock"]),
                "product_url": f"https://prod.danawa.com/info/?pcode={random.randint(10000000, 99999999)}",
            }

            logger.info(f"✓ {gpu_model}: {price:,}원")
            return result

        except Exception as e:
            logger.error(f"✗ {gpu_model} 크롤링 실패: {e}")
            return None

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
