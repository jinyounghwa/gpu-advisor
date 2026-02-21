"""
GPU 뉴스 크롤러
Google News RSS에서 GPU 관련 뉴스 수집 및 키워드 기반 감성 분석
"""
import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 감성 키워드
POSITIVE_KEYWORDS = [
    "price drop", "cheaper", "discount", "sale", "deal", "lower price",
    "price cut", "price fall", "affordable", "budget", "value",
    "in stock", "available", "launch", "release", "new",
    "가격 인하", "할인", "재고", "출시",
]
NEGATIVE_KEYWORDS = [
    "price rise", "price hike", "expensive", "shortage", "out of stock",
    "tariff", "tax", "ban", "supply chain", "delay", "recall",
    "higher price", "increase", "inflation", "scalper", "sold out",
    "품절", "부족", "가격 인상", "관세",
]


def _score_sentiment(text: str) -> tuple[str, float]:
    """키워드 기반 감성 점수 계산"""
    text_lower = text.lower()
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)

    if pos == neg == 0:
        return "neutral", 0.0
    total = pos + neg
    score = (pos - neg) / total  # [-1, 1]
    if score > 0.1:
        return "positive", round(score, 2)
    elif score < -0.1:
        return "negative", round(score, 2)
    return "neutral", round(score, 2)


class NewsCrawler:
    """GPU 뉴스 크롤러 (Google News RSS)"""

    RSS_BASE = "https://news.google.com/rss/search"

    def __init__(self, output_dir: str = "data/raw/news"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.keywords = [
            "NVIDIA GPU price",
            "AMD Radeon GPU",
            "Intel Arc GPU",
            "graphics card price",
            "GPU shortage",
            "RTX 5090",
            "RTX 5080",
            "GPU stock",
        ]

    def fetch_news(self, keyword: str) -> list:
        """Google News RSS에서 키워드 기사 수집"""
        params = {
            "q": keyword,
            "hl": "en",
            "gl": "US",
            "ceid": "US:en",
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(
            self.RSS_BASE, params=params, headers=headers, timeout=10
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")

        articles = []
        for item in items[:5]:  # 키워드당 최대 5개
            title_el = item.find("title")
            link_el = item.find("link")
            pub_el = item.find("pubDate")
            source_el = item.find("source")

            if not title_el:
                continue

            title = title_el.get_text(strip=True)
            link = link_el.get_text(strip=True) if link_el else ""
            pub_date = pub_el.get_text(strip=True) if pub_el else ""
            source = source_el.get_text(strip=True) if source_el else ""

            sentiment, score = _score_sentiment(title)

            articles.append({
                "title": title,
                "url": link,
                "source": source,
                "published_at": pub_date,
                "sentiment": sentiment,
                "sentiment_score": score,
                "keywords": [keyword],
            })

        logger.info(f"  {keyword}: {len(articles)}개 기사")
        return articles

    def crawl_all(self) -> list:
        """모든 키워드로 뉴스 수집"""
        logger.info("=" * 80)
        logger.info(f"GPU 뉴스 크롤링 시작 - {datetime.now()}")
        logger.info("=" * 80)

        all_articles = []
        seen_titles = set()

        for keyword in self.keywords:
            try:
                articles = self.fetch_news(keyword)
                for a in articles:
                    if a["title"] not in seen_titles:
                        seen_titles.add(a["title"])
                        all_articles.append(a)
            except Exception as e:
                logger.warning(f"  ⚠ {keyword}: {e}")
            time.sleep(0.5)

        logger.info(f"\n✓ 총 {len(all_articles)}개 기사 수집 (중복 제거)")
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
        return {
            "total": len(articles),
            "sentiment_avg": round(sum(sentiments) / len(sentiments), 2),
            "positive_count": sum(1 for a in articles if a["sentiment"] == "positive"),
            "negative_count": sum(1 for a in articles if a["sentiment"] == "negative"),
            "neutral_count": sum(1 for a in articles if a["sentiment"] == "neutral"),
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
