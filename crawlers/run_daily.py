#!/usr/bin/env python3
"""
일일 크롤링 실행 스크립트
모든 크롤러를 순차적으로 실행하고 Feature 생성
"""
import sys
from pathlib import Path
from datetime import datetime
import logging

# 프로젝트 루트를 Python path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from crawlers.danawa_crawler import DanawaCrawler
from crawlers.exchange_rate_crawler import ExchangeRateCrawler
from crawlers.news_crawler import NewsCrawler
from crawlers.feature_engineer import FeatureEngineer

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/daily_crawl.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """메인 실행 함수"""
    logger.info("=" * 80)
    logger.info(f"일일 데이터 수집 시작 - {datetime.now()}")
    logger.info("=" * 80)

    try:
        # 1. 다나와 GPU 가격 크롤링
        logger.info("\n[1/4] 다나와 GPU 가격 크롤링")
        danawa = DanawaCrawler()
        products = danawa.crawl_all()
        danawa.save(products)

        # 2. 환율 크롤링
        logger.info("\n[2/4] 환율 정보 수집")
        exchange = ExchangeRateCrawler()
        exchange.run()

        # 3. 뉴스 크롤링
        logger.info("\n[3/4] GPU 뉴스 크롤링")
        news = NewsCrawler()
        news.run()

        # 4. Feature Engineering (256차원 생성)
        logger.info("\n[4/4] Feature Engineering (256차원)")
        engineer = FeatureEngineer()
        output_file = engineer.process_all()

        logger.info("\n" + "=" * 80)
        logger.info("✓ 일일 데이터 수집 완료!")
        logger.info(f"✓ Feature 파일: {output_file}")
        logger.info(f"✓ 시간: {datetime.now()}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"\n✗ 오류 발생: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
