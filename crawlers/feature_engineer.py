"""
Feature Engineering: 11차원 → 256차원 확장
GPU 구매 타이밍 예측을 위한 고차원 Feature 생성
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """256차원 Feature 생성기"""

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_historical_data(self, days: int = 30) -> dict:
        """과거 N일 데이터 로드"""
        historical_data = defaultdict(lambda: {"prices": [], "dates": []})

        # 다나와 데이터
        danawa_dir = self.raw_data_dir / "danawa"
        if danawa_dir.exists():
            for file in sorted(danawa_dir.glob("*.json"))[-days:]:
                with open(file) as f:
                    data = json.load(f)
                    for product in data.get("products", []):
                        gpu_model = product["chipset"]
                        historical_data[gpu_model]["prices"].append(
                            product["lowest_price"]
                        )
                        historical_data[gpu_model]["dates"].append(data["date"])

        return historical_data

    def calculate_price_features(
        self, current_price: float, historical_prices: list
    ) -> list:
        """
        가격 관련 Feature (60차원)
        """
        features = []

        # 기본 가격 정보
        features.append(current_price / 10000000)  # 정규화 (0~1)

        if len(historical_prices) >= 7:
            # 이동평균
            ma_7d = np.mean(historical_prices[-7:])
            features.append(ma_7d / 10000000)
        else:
            features.append(0.5)

        if len(historical_prices) >= 14:
            ma_14d = np.mean(historical_prices[-14:])
            features.append(ma_14d / 10000000)
        else:
            features.append(0.5)

        if len(historical_prices) >= 30:
            ma_30d = np.mean(historical_prices[-30:])
            features.append(ma_30d / 10000000)
        else:
            features.append(0.5)

        # 가격 변화율
        if len(historical_prices) >= 2:
            change_1d = (current_price - historical_prices[-2]) / historical_prices[-2]
            features.append(change_1d)
        else:
            features.append(0.0)

        if len(historical_prices) >= 7:
            change_7d = (current_price - historical_prices[-7]) / historical_prices[-7]
            features.append(change_7d)
        else:
            features.append(0.0)

        # 변동성 (표준편차)
        if len(historical_prices) >= 7:
            volatility = np.std(historical_prices[-7:]) / np.mean(historical_prices[-7:])
            features.append(volatility)
        else:
            features.append(0.0)

        # 최고가/최저가
        if len(historical_prices) >= 7:
            price_max = max(historical_prices[-7:])
            price_min = min(historical_prices[-7:])
            features.append(price_max / 10000000)
            features.append(price_min / 10000000)
            features.append((price_max - price_min) / price_max)  # 변동 폭
        else:
            features.extend([0.5, 0.5, 0.0])

        # 추세 (상승/하락)
        if len(historical_prices) >= 7:
            trend = (
                np.mean(historical_prices[-3:]) - np.mean(historical_prices[-7:-3])
            ) / np.mean(historical_prices[-7:])
            features.append(trend)
        else:
            features.append(0.0)

        # 패딩 (60차원 맞추기)
        while len(features) < 60:
            features.append(0.0)

        return features[:60]

    def calculate_exchange_features(self, exchange_data: dict) -> list:
        """
        환율 관련 Feature (20차원)
        """
        features = []

        usd_krw = exchange_data.get("USD/KRW", 1400) / 2000  # 정규화
        jpy_krw = exchange_data.get("JPY/KRW", 900) / 1500
        eur_krw = exchange_data.get("EUR/KRW", 1500) / 2000

        features.extend([usd_krw, jpy_krw, eur_krw])

        # 패딩
        while len(features) < 20:
            features.append(0.5)

        return features[:20]

    def calculate_news_features(self, news_stats: dict) -> list:
        """
        뉴스 감정 분석 Feature (30차원)
        """
        features = []

        sentiment_avg = news_stats.get("sentiment_avg", 0.0)
        total_articles = news_stats.get("total", 0)
        positive_count = news_stats.get("positive_count", 0)
        negative_count = news_stats.get("negative_count", 0)

        features.extend(
            [
                (sentiment_avg + 1) / 2,  # -1~1 → 0~1
                min(total_articles / 100, 1.0),  # 정규화
                min(positive_count / 50, 1.0),
                min(negative_count / 50, 1.0),
            ]
        )

        # 패딩
        while len(features) < 30:
            features.append(0.0)

        return features[:30]

    def calculate_market_features(self, product_info: dict) -> list:
        """
        시장 공급 Feature (20차원)
        """
        features = []

        seller_count = product_info.get("seller_count", 0)
        features.append(min(seller_count / 50, 1.0))

        stock_status_map = {"in_stock": 1.0, "low_stock": 0.5, "out_of_stock": 0.0}
        stock_status = product_info.get("stock_status", "in_stock")
        features.append(stock_status_map.get(stock_status, 0.5))

        # 패딩
        while len(features) < 20:
            features.append(0.5)

        return features[:20]

    def calculate_time_features(self, date_str: str) -> list:
        """
        시간 관련 Feature (20차원)
        """
        features = []

        date = datetime.strptime(date_str, "%Y-%m-%d")

        # 요일 (0=월요일, 6=일요일)
        day_of_week = date.weekday() / 7
        features.append(day_of_week)

        # 월
        month = date.month / 12
        features.append(month)

        # 연말 여부 (12월)
        is_year_end = 1.0 if date.month == 12 else 0.0
        features.append(is_year_end)

        # 패딩
        while len(features) < 20:
            features.append(0.0)

        return features[:20]

    def calculate_technical_indicators(self, historical_prices: list) -> list:
        """
        기술적 지표 Feature (106차원)
        """
        features = []

        if len(historical_prices) >= 14:
            # RSI (Relative Strength Index)
            gains = []
            losses = []
            for i in range(1, len(historical_prices)):
                diff = historical_prices[i] - historical_prices[i - 1]
                if diff > 0:
                    gains.append(diff)
                else:
                    losses.append(abs(diff))

            avg_gain = np.mean(gains[-14:]) if gains else 0
            avg_loss = np.mean(losses[-14:]) if losses else 0

            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            features.append(rsi / 100)  # 0~1 정규화
        else:
            features.append(0.5)

        # MACD (Moving Average Convergence Divergence)
        if len(historical_prices) >= 26:
            ema_12 = np.mean(historical_prices[-12:])
            ema_26 = np.mean(historical_prices[-26:])
            macd = (ema_12 - ema_26) / ema_26
            features.append(macd)
        else:
            features.append(0.0)

        # 모멘텀
        if len(historical_prices) >= 10:
            momentum = (
                historical_prices[-1] - historical_prices[-10]
            ) / historical_prices[-10]
            features.append(momentum)
        else:
            features.append(0.0)

        # 패딩
        while len(features) < 106:
            features.append(0.0)

        return features[:106]

    def generate_features(self, gpu_model: str, date_str: str) -> np.ndarray:
        """
        특정 GPU 모델의 256차원 Feature 생성
        """
        # 과거 데이터 로드
        historical_data = self.load_historical_data(days=30)
        historical_prices = historical_data.get(gpu_model, {}).get("prices", [])

        # 오늘 데이터 로드
        today_file = self.raw_data_dir / "danawa" / f"{date_str}.json"
        if not today_file.exists():
            logger.warning(f"데이터 파일 없음: {today_file}")
            return np.zeros(256)

        with open(today_file) as f:
            today_data = json.load(f)

        # GPU 제품 찾기
        product_info = None
        for product in today_data.get("products", []):
            if product["chipset"] == gpu_model:
                product_info = product
                break

        if not product_info:
            logger.warning(f"제품 정보 없음: {gpu_model}")
            return np.zeros(256)

        current_price = product_info["lowest_price"]

        # 환율 데이터
        exchange_file = self.raw_data_dir / "exchange" / f"{date_str}.json"
        exchange_data = {}
        if exchange_file.exists():
            with open(exchange_file) as f:
                exchange_json = json.load(f)
                exchange_data = exchange_json.get("rates", {})

        # 뉴스 데이터
        news_file = self.raw_data_dir / "news" / f"{date_str}.json"
        news_stats = {}
        if news_file.exists():
            with open(news_file) as f:
                news_json = json.load(f)
                news_stats = news_json.get("statistics", {})

        # Feature 생성
        features = []

        # 1. 가격 Feature (60차원)
        features.extend(self.calculate_price_features(current_price, historical_prices))

        # 2. 환율 Feature (20차원)
        features.extend(self.calculate_exchange_features(exchange_data))

        # 3. 뉴스 Feature (30차원)
        features.extend(self.calculate_news_features(news_stats))

        # 4. 시장 Feature (20차원)
        features.extend(self.calculate_market_features(product_info))

        # 5. 시간 Feature (20차원)
        features.extend(self.calculate_time_features(date_str))

        # 6. 기술 지표 (106차원)
        features.extend(self.calculate_technical_indicators(historical_prices))

        # 256차원 맞추기
        features = features[:256]
        while len(features) < 256:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    def process_all(self, date_str: str = None) -> str:
        """모든 GPU 모델 Feature 생성"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 80)
        logger.info(f"Feature Engineering 시작 - {date_str}")
        logger.info("=" * 80)

        # GPU 모델 리스트
        danawa_file = self.raw_data_dir / "danawa" / f"{date_str}.json"
        if not danawa_file.exists():
            logger.error(f"데이터 파일 없음: {danawa_file}")
            return None

        with open(danawa_file) as f:
            today_data = json.load(f)

        gpu_models = [p["chipset"] for p in today_data.get("products", [])]

        # Feature 생성
        training_data = []

        for gpu_model in gpu_models:
            logger.info(f"처리 중: {gpu_model}")
            feature_vector = self.generate_features(gpu_model, date_str)

            training_data.append(
                {
                    "date": date_str,
                    "gpu_model": gpu_model,
                    "state_vector": feature_vector.tolist(),
                }
            )

        # 저장
        output_dir = self.processed_dir / "dataset"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"training_data_{date_str}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✓ Feature 생성 완료: {len(training_data)}개 샘플")
        logger.info(f"✓ 저장 위치: {output_file}")
        logger.info(f"✓ Feature 차원: 256")

        return str(output_file)


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.process_all()
