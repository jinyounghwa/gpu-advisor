"""
환율 크롤러
USD/KRW, JPY/KRW, EUR/KRW 수집
출처: open.er-api.com (무료, 인증 불필요)
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

    API_URL = "https://open.er-api.com/v6/latest/USD"

    def __init__(self, output_dir: str = "data/raw/exchange"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fetch_exchange_rates(self) -> dict:
        """
        환율 정보 가져오기
        open.er-api.com에서 USD 기준 실시간 환율 수집
        """
        response = requests.get(self.API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("result") != "success":
            raise ValueError(f"API 오류: {data}")

        rates = data["rates"]

        # KRW 기준으로 변환
        usd_krw = rates["KRW"]
        jpy_krw = rates["KRW"] / rates["JPY"]   # 100엔당 원
        eur_krw = rates["KRW"] / rates["EUR"]

        result = {
            "USD/KRW": round(usd_krw, 2),
            "JPY/KRW": round(jpy_krw * 100, 2),  # 100엔 기준
            "EUR/KRW": round(eur_krw, 2),
        }

        logger.info("환율 정보 수집 완료:")
        for currency, rate in result.items():
            logger.info(f"  {currency}: {rate:,.2f}")

        return result

    def save(self, data: dict) -> str:
        """환율 데이터 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.output_dir / f"{today}.json"

        output_data = {
            "date": today,
            "source": "open.er-api.com",
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

        try:
            rates = self.fetch_exchange_rates()
            self.save(rates)
        except Exception as e:
            logger.error(f"✗ 환율 크롤링 실패: {e}")
            raise


if __name__ == "__main__":
    crawler = ExchangeRateCrawler()
    crawler.run()
