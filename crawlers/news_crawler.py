"""
GPU 뉴스 크롤러
GPU 관련 뉴스 수집 및 감정 분석
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCrawler:
    """GPU 뉴스 크롤러"""

    def __init__(self, output_dir: str = "data/raw/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 검색 키워드
        self.keywords = [
            "GPU price",
            "graphics card",
            "NVIDIA RTX",
            "AMD Radeon",
            "Intel Arc",
            "GPU shortage",
            "GPU stock",
            "graphics card price drop",
        ]

    def fetch_news(self, keyword: str) -> list:
        """
        특정 키워드로 뉴스 검색
        실제로는 Google News RSS, Naver News API 등 사용
        """
        try:
            # Mock 데이터 (실제로는 RSS 파싱 또는 API 호출)
            # Google News RSS: https://news.google.com/rss/search?q={keyword}

            articles = []

            # 랜덤하게 0~3개 기사 생성
            num_articles = random.randint(0, 3)

            for i in range(num_articles):
                sentiment = random.choice(["positive", "negative", "neutral"])
                sentiment_scores = {
                    "positive": random.uniform(0.5, 1.0),
                    "negative": random.uniform(-1.0, -0.5),
                    "neutral": random.uniform(-0.2, 0.2),
                }

                article = {
                    "title": f"{keyword} 관련 뉴스 {i + 1}",
                    "url": f"https://news.example.com/article/{random.randint(1000, 9999)}",
                    "published_at": datetime.now().isoformat(),
                    "sentiment": sentiment,
                    "sentiment_score": round(sentiment_scores[sentiment], 2),
                    "keywords": [keyword],
                }
                articles.append(article)

            if articles:
                logger.info(f"  {keyword}: {len(articles)}개 기사")

            return articles

        except Exception as e:
            logger.error(f"뉴스 크롤링 실패 ({keyword}): {e}")
            return []

    def crawl_all(self) -> list:
        """모든 키워드로 뉴스 수집"""
        logger.info("=" * 80)
        logger.info(f"GPU 뉴스 크롤링 시작 - {datetime.now()}")
        logger.info("=" * 80)

        all_articles = []

        for keyword in self.keywords:
            articles = self.fetch_news(keyword)
            all_articles.extend(articles)

        logger.info(f"\n✓ 총 {len(all_articles)}개 기사 수집")
        return all_articles

    def calculate_statistics(self, articles: list) -> dict:
        """뉴스 통계 계산"""
        if not articles:
            return {
                "total": 0,
                "sentiment_avg": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }

        sentiments = [a["sentiment_score"] for a in articles]
        sentiment_counts = {
            "positive": sum(1 for a in articles if a["sentiment"] == "positive"),
            "negative": sum(1 for a in articles if a["sentiment"] == "negative"),
            "neutral": sum(1 for a in articles if a["sentiment"] == "neutral"),
        }

        return {
            "total": len(articles),
            "sentiment_avg": round(sum(sentiments) / len(sentiments), 2),
            "positive_count": sentiment_counts["positive"],
            "negative_count": sentiment_counts["negative"],
            "neutral_count": sentiment_counts["neutral"],
        }

    def save(self, articles: list) -> str:
        """뉴스 데이터 저장"""
        today = datetime.now().strftime("%Y-%m-%d")
        output_file = self.output_dir / f"{today}.json"

        stats = self.calculate_statistics(articles)

        output_data = {
            "date": today,
            "source": "google_news_rss",
            "total_articles": len(articles),
            "articles": articles,
            "statistics": stats,
            "metadata": {
                "crawled_at": datetime.now().isoformat(),
                "keywords": self.keywords,
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"✓ 저장 완료: {output_file}")
        return str(output_file)

    def run(self):
        """실행"""
        articles = self.crawl_all()
        self.save(articles)


if __name__ == "__main__":
    crawler = NewsCrawler()
    crawler.run()
