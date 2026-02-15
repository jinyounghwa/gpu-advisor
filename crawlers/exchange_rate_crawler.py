"""
환율 크롤러
USD/KRW, JPY/KRW, EUR/KRW 수집
"""
import requests
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangeRateCrawler:
    """환율 크롤러"""

    def __init__(self, output_dir: str = "data/raw/exchange"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_exchange_rates(self) -> dict:
        """
        환율 정보 가져오기
        실제로는 한국은행 API, exchangerate-api.com 등 사용
        """
        try:
            # Mock 데이터 (실제로는 API 호출)
            # API 예시: https://api.exchangerate-api.com/v4/latest/USD
            # response = requests.get(api_url, timeout=10)
            # data = response.json()

            import random

            # 기준 환율에서 ±1% 변동
            base_usd_krw = 1442.7
            base_jpy_krw = 943.28
            base_eur_krw = 1560.5

            rates = {
                "USD/KRW": round(base_usd_krw * (1 + random.uniform(-0.01, 0.01)), 2),
                "JPY/KRW": round(base_jpy_krw * (1 + random.uniform(-0.01, 0.01)), 2),
                "EUR/KRW": round(base_eur_krw * (1 + random.uniform(-0.01, 0.01)), 2),
            }

            logger.info("환율 정보 수집 완료:")
            for currency, rate in rates.items():
                logger.info(f"  {currency}: {rate}")

            return rates

        except Exception as e:
            logger.error(f"환율 크롤링 실패: {e}")
            return {}

    def save(self, data: dict) -> str:
        """환율 데이터 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.output_dir / f"{today}.json"

        output_data = {
            "date": today,
            "source": "exchange_rate_api",
            "rates": data,
            "metadata": {"fetched_at": datetime.now().isoformat()},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 저장 완료: {output_file}")
        return str(output_file)

    def run(self):
        """실행"""
        logger.info("=" * 80)
        logger.info(f"환율 정보 수집 시작 - {datetime.now()}")
        logger.info("=" * 80)

        rates = self.fetch_exchange_rates()
        if rates:
            self.save(rates)


if __name__ == "__main__":
    crawler = ExchangeRateCrawler()
    crawler.run()
