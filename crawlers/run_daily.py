#!/usr/bin/env python3
"""
일일 크롤링 실행 스크립트
모든 크롤러를 순차적으로 실행하고 Feature 생성
"""
import sys
import traceback
from pathlib import Path
from datetime import datetime
import logging

# 프로젝트 루트 절대경로 (cron 환경에서도 안전)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

# 로그 디렉토리 절대경로로 보장
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# 크롤러 모듈 import 전에 루트 로거 설정 (basicConfig 선점)
_root_logger = logging.getLogger()
_root_logger.setLevel(logging.INFO)
_fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
_file_handler = logging.FileHandler(LOG_DIR / "daily_crawl.log")
_file_handler.setFormatter(_fmt)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_fmt)
_root_logger.addHandler(_file_handler)
_root_logger.addHandler(_stream_handler)

from crawlers.danawa_crawler import DanawaCrawler
from crawlers.exchange_rate_crawler import ExchangeRateCrawler
from crawlers.news_crawler import NewsCrawler
from crawlers.feature_engineer import FeatureEngineer
from crawlers.status_report import generate_daily_status_report
logger = logging.getLogger(__name__)


def run_step(name: str, fn):
    """단계 실행 — 실패해도 다음 단계 계속"""
    logger.info(f"\n{name}")
    try:
        fn()
        return True
    except Exception as e:
        logger.error(f"✗ {name} 실패: {e}")
        traceback.print_exc()
        return False


def main():
    """메인 실행 함수"""
    run_started_at = datetime.now().isoformat()
    logger.info("=" * 80)
    logger.info(f"일일 데이터 수집 시작 - {run_started_at}")
    logger.info(f"프로젝트 경로: {PROJECT_ROOT}")
    logger.info("=" * 80)

    results = {}

    # 1. 다나와 GPU 가격 크롤링
    def step_danawa():
        danawa = DanawaCrawler(str(PROJECT_ROOT / "data/raw/danawa"))
        products = danawa.crawl_all()
        danawa.save(products)

    results["danawa"] = run_step("[1/4] 다나와 GPU 가격 크롤링", step_danawa)

    # 2. 환율 수집
    def step_exchange():
        exchange = ExchangeRateCrawler(str(PROJECT_ROOT / "data/raw/exchange"))
        exchange.run()

    results["exchange"] = run_step("[2/4] 환율 정보 수집", step_exchange)

    # 3. 뉴스 크롤링
    def step_news():
        news = NewsCrawler(str(PROJECT_ROOT / "data/raw/news"))
        news.run()

    results["news"] = run_step("[3/4] GPU 뉴스 크롤링", step_news)

    # 4. Feature Engineering (danawa 성공 시에만)
    feature_file = None
    if results["danawa"]:
        def step_feature():
            nonlocal feature_file
            engineer = FeatureEngineer(
                raw_data_dir=str(PROJECT_ROOT / "data/raw"),
                processed_dir=str(PROJECT_ROOT / "data/processed"),
            )
            feature_file = engineer.process_all()

        results["feature"] = run_step("[4/4] Feature Engineering (256차원)", step_feature)
    else:
        logger.warning("[4/4] 다나와 수집 실패로 Feature Engineering 건너뜀")
        results["feature"] = False

    # 결과 요약
    logger.info("\n" + "=" * 80)
    ok = all(results.values())
    logger.info("✓ 일일 데이터 수집 완료!" if ok else "⚠ 일부 단계 실패")
    for step, success in results.items():
        logger.info(f"  {'✓' if success else '✗'} {step}")
    if feature_file:
        logger.info(f"  Feature 파일: {feature_file}")

    try:
        reports = generate_daily_status_report(
            project_root=PROJECT_ROOT,
            step_results=results,
            feature_file=feature_file,
            run_started_at=run_started_at,
        )
        logger.info(f"  상태 리포트(JSON): {reports['json_report']}")
        logger.info(f"  상태 리포트(MD): {reports['markdown_report']}")
        logger.info(f"  최신 리포트(JSON): {reports['latest_json']}")
        logger.info(f"  최신 리포트(MD): {reports['latest_markdown']}")
    except Exception as e:
        logger.error(f"상태 리포트 생성 실패: {e}")

    logger.info(f"  완료 시각: {datetime.now()}")
    logger.info("=" * 80)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
