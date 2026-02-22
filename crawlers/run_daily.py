#!/usr/bin/env python3
"""
일일 크롤링 실행 스크립트
모든 크롤러를 순차적으로 실행하고 Feature 생성
"""
import argparse
import fcntl
import sys
import traceback
from pathlib import Path
from datetime import datetime
import logging
import time
import resource

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


class SingleRunLock:
    """중복 실행 방지용 파일 락."""

    def __init__(self, lock_path: Path):
        self.lock_path = lock_path
        self._fp = None

    def acquire(self) -> bool:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.lock_path, "w", encoding="utf-8")
        try:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        self._fp.write(str(datetime.now().isoformat()))
        self._fp.flush()
        return True

    def release(self) -> None:
        if not self._fp:
            return
        try:
            fcntl.flock(self._fp.fileno(), fcntl.LOCK_UN)
        finally:
            self._fp.close()
            self._fp = None


def _max_rss_mb() -> float:
    # macOS ru_maxrss unit is bytes, Linux is KB.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / 1024.0 / 1024.0
    return maxrss / 1024.0


def run_step(name: str, fn):
    """단계 실행 — 실패해도 다음 단계 계속"""
    logger.info(f"\n{name}")
    started = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - started
        logger.info(f"✓ {name} 완료 ({elapsed:.2f}s)")
        return True
    except Exception as e:
        logger.error(f"✗ {name} 실패: {e}")
        traceback.print_exc()
        elapsed = time.perf_counter() - started
        logger.info(f"✗ {name} 종료 ({elapsed:.2f}s)")
        return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GPU Advisor daily crawler runner")
    parser.add_argument(
        "--skip-release",
        action="store_true",
        help="Skip backend release pipeline stage to reduce runtime and memory usage.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    """메인 실행 함수"""
    args = _parse_args(argv)
    run_started_perf = time.perf_counter()
    run_started_at = datetime.now().isoformat()
    logger.info("=" * 80)
    logger.info(f"일일 데이터 수집 시작 - {run_started_at}")
    logger.info(f"프로젝트 경로: {PROJECT_ROOT}")
    logger.info(f"옵션: skip_release={args.skip_release}")
    logger.info("=" * 80)

    run_lock = SingleRunLock(LOG_DIR / "daily_crawl.lock")
    if not run_lock.acquire():
        logger.error("이미 실행 중인 프로세스가 있어 종료합니다. (lock: logs/daily_crawl.lock)")
        return 2

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

    try:
        if args.skip_release:
            logger.info("  릴리즈 파이프라인 생략(--skip-release)")
        else:
            # 일일 배치에서는 긴 학습 없이도 release 리포트를 항상 남긴다.
            # (데이터 부족/게이트 실패 시에도 blocked 리포트가 생성됨)
            from backend.agent import AgentReleasePipeline, PipelineConfig

            pipeline = AgentReleasePipeline(project_root=PROJECT_ROOT)
            release_cfg = PipelineConfig(
                run_training=False,
                require_30d=False,
                lookback_days=30,
            )
            release_result = pipeline.run(release_cfg)
            release_reports = release_result.get("reports", {})
            logger.info(f"  릴리즈 판정: {release_result.get('status')}")
            logger.info(f"  릴리즈 리포트(JSON): {release_reports.get('json_report')}")
            logger.info(f"  릴리즈 리포트(MD): {release_reports.get('markdown_report')}")
            logger.info(f"  최신 릴리즈(JSON): {release_reports.get('latest_json')}")
            logger.info(f"  최신 릴리즈(MD): {release_reports.get('latest_markdown')}")
    except Exception as e:
        logger.error(f"릴리즈 리포트 생성 실패: {e}")
    finally:
        elapsed_total = time.perf_counter() - run_started_perf
        logger.info(f"  완료 시각: {datetime.now()}")
        logger.info(f"  총 실행시간: {elapsed_total:.2f}s")
        logger.info(f"  프로세스 Peak RSS: {_max_rss_mb():.1f} MB")
        logger.info("=" * 80)
        run_lock.release()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
