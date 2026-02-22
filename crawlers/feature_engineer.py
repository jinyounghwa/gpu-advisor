"""
Feature Engineering: 실측 지표 우선 256차원 생성기

원칙:
- 가능한 한 실제 수집 데이터에서 계산
- 계산 불가능한 항목은 임의 상수 대신 value=0.0 + missing_mask=0.0으로 표기
- 총 차원은 기존과 동일하게 256 유지
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    return float(numer / denom) if abs(denom) > 1e-12 else float(default)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def _ema(arr: np.ndarray, span: int) -> tuple[float, float]:
    if arr.size < span:
        return 0.0, 0.0
    alpha = 2.0 / (span + 1.0)
    out = float(arr[0])
    for v in arr[1:]:
        out = alpha * float(v) + (1.0 - alpha) * out
    return out, 1.0


def _rsi(arr: np.ndarray, period: int) -> tuple[float, float]:
    if arr.size < period + 1:
        return 0.0, 0.0
    diffs = np.diff(arr)
    gains = np.where(diffs > 0, diffs, 0.0)
    losses = np.where(diffs < 0, -diffs, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss <= 1e-12:
        return 1.0, 1.0
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi / 100.0), 1.0


def _window(arr: np.ndarray, n: int) -> tuple[np.ndarray, float]:
    if arr.size < n:
        return np.asarray([], dtype=np.float64), 0.0
    return arr[-n:], 1.0


def _returns(arr: np.ndarray) -> np.ndarray:
    if arr.size < 2:
        return np.asarray([], dtype=np.float64)
    prev = arr[:-1]
    cur = arr[1:]
    out = np.zeros_like(cur, dtype=np.float64)
    nz = np.abs(prev) > 1e-12
    out[nz] = (cur[nz] - prev[nz]) / prev[nz]
    return out


class FeatureEngineer:
    """256차원 Feature 생성기 (실측 지표 + 결측 마스크)."""

    def __init__(
        self,
        raw_data_dir: str = "data/raw",
        processed_dir: str = "data/processed",
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def load_historical_data(self, days: int = 30) -> dict[str, list[float]]:
        """과거 N일 다나와 데이터를 모델별 price 시계열로 로드."""
        historical_data: dict[str, list[float]] = defaultdict(list)
        danawa_dir = self.raw_data_dir / "danawa"
        if danawa_dir.exists():
            for file in sorted(danawa_dir.glob("*.json"))[-days:]:
                with open(file, encoding="utf-8") as f:
                    data = json.load(f)
                for product in data.get("products", []):
                    gpu_model = product["chipset"]
                    historical_data[gpu_model].append(float(product["lowest_price"]))
        return dict(historical_data)

    def _build_feature_context(self, date_str: str) -> dict | None:
        """특정 날짜의 feature 생성에 필요한 입력을 한 번만 로드."""
        today_file = self.raw_data_dir / "danawa" / f"{date_str}.json"
        if not today_file.exists():
            logger.warning(f"데이터 파일 없음: {today_file}")
            return None

        with open(today_file, encoding="utf-8") as f:
            today_data = json.load(f)
        today_products = today_data.get("products", [])
        product_by_model = {p.get("chipset"): p for p in today_products if p.get("chipset")}

        exchange_file = self.raw_data_dir / "exchange" / f"{date_str}.json"
        exchange_data = {}
        if exchange_file.exists():
            with open(exchange_file, encoding="utf-8") as f:
                exchange_data = json.load(f).get("rates", {})

        news_file = self.raw_data_dir / "news" / f"{date_str}.json"
        news_stats = {}
        today_articles: list[dict] = []
        if news_file.exists():
            with open(news_file, encoding="utf-8") as f:
                news_json = json.load(f)
            news_stats = news_json.get("statistics", {})
            today_articles = news_json.get("articles", [])

        return {
            "historical_data": self.load_historical_data(days=30),
            "today_products": today_products,
            "product_by_model": product_by_model,
            "exchange_data": exchange_data,
            "exchange_history": self._load_exchange_history(days=30),
            "news_stats": news_stats,
            "today_articles": today_articles,
            "news_history": self._load_news_history(days=30),
        }

    def _load_exchange_history(self, days: int = 30) -> list[dict]:
        rows: list[dict] = []
        root = self.raw_data_dir / "exchange"
        if not root.exists():
            return rows
        for file in sorted(root.glob("*.json"))[-days:]:
            with open(file, encoding="utf-8") as f:
                d = json.load(f)
            rates = d.get("rates", {})
            rows.append(
                {
                    "date": d.get("date", file.stem),
                    "USD/KRW": float(rates.get("USD/KRW", 0.0)),
                    "JPY/KRW": float(rates.get("JPY/KRW", 0.0)),
                    "EUR/KRW": float(rates.get("EUR/KRW", 0.0)),
                }
            )
        return rows

    def _load_news_history(self, days: int = 30) -> list[dict]:
        rows: list[dict] = []
        root = self.raw_data_dir / "news"
        if not root.exists():
            return rows
        for file in sorted(root.glob("*.json"))[-days:]:
            with open(file, encoding="utf-8") as f:
                d = json.load(f)
            s = d.get("statistics", {})
            rows.append(
                {
                    "date": d.get("date", file.stem),
                    "sentiment_avg": float(s.get("sentiment_avg", 0.0)),
                    "total": float(s.get("total", 0.0)),
                    "positive_count": float(s.get("positive_count", 0.0)),
                    "negative_count": float(s.get("negative_count", 0.0)),
                }
            )
        return rows

    def calculate_price_features(self, current_price: float, historical_prices: list) -> list:
        """가격 Feature (60차원) - 실측 51 + 가용성 마스크 9."""
        arr = np.asarray(historical_prices, dtype=np.float64)
        vals: list[float] = []

        # 기존 의미 보존 구간 (0~10)
        vals.append(float(current_price) / 10_000_000.0)  # 0
        w7, m7 = _window(arr, 7)
        vals.append(float(np.mean(w7)) / 10_000_000.0 if m7 else 0.0)  # 1
        w14, m14 = _window(arr, 14)
        vals.append(float(np.mean(w14)) / 10_000_000.0 if m14 else 0.0)  # 2
        w30, m30 = _window(arr, 30)
        vals.append(float(np.mean(w30)) / 10_000_000.0 if m30 else 0.0)  # 3
        vals.append(_safe_div(current_price - arr[-2], arr[-2]) if arr.size >= 2 else 0.0)  # 4
        vals.append(_safe_div(current_price - arr[-7], arr[-7]) if arr.size >= 7 else 0.0)  # 5
        vals.append(_safe_div(np.std(w7), np.mean(w7)) if m7 else 0.0)  # 6
        vals.append(float(np.max(w7)) / 10_000_000.0 if m7 else 0.0)  # 7
        vals.append(float(np.min(w7)) / 10_000_000.0 if m7 else 0.0)  # 8
        vals.append(_safe_div(np.max(w7) - np.min(w7), np.max(w7)) if m7 else 0.0)  # 9
        if arr.size >= 7:
            vals.append(_safe_div(np.mean(arr[-3:]) - np.mean(arr[-7:-3]), np.mean(arr[-7:])))  # 10
        else:
            vals.append(0.0)

        # 추가 실측 지표 (11~50)
        for lag in [2, 3, 5, 10, 14, 21, 30]:
            vals.append(_safe_div(arr[-1] - arr[-lag], arr[-lag]) if arr.size >= lag else 0.0)

        for win in [3, 5, 7, 14, 21, 30]:
            if arr.size >= win:
                ma = float(np.mean(arr[-win:]))
                vals.append(_safe_div(arr[-1] - ma, ma))
            else:
                vals.append(0.0)

        for win in [3, 5, 7, 14, 21, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                vals.append(_safe_div(np.std(seg), np.mean(seg)))
            else:
                vals.append(0.0)

        for win in [5, 7, 14, 21, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                vals.append(_safe_div(arr[-1] - np.min(seg), np.max(seg) - np.min(seg)))
            else:
                vals.append(0.0)

        for win in [7, 14, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                vals.append(_safe_div(arr[-1] - np.mean(seg), np.std(seg)))
            else:
                vals.append(0.0)

        for win in [5, 10, 20, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                x = np.arange(win, dtype=np.float64)
                slope = np.polyfit(x, seg, 1)[0]
                vals.append(_safe_div(slope, np.mean(seg)))
            else:
                vals.append(0.0)

        if arr.size >= 5:
            vals.extend(
                [
                    float(np.quantile(arr, 0.25)) / 10_000_000.0,
                    float(np.quantile(arr, 0.50)) / 10_000_000.0,
                    float(np.quantile(arr, 0.75)) / 10_000_000.0,
                ]
            )
        else:
            vals.extend([0.0, 0.0, 0.0])

        if arr.size >= 2:
            peak = np.maximum.accumulate(arr)
            drawdown = (arr - peak) / np.maximum(peak, 1.0)
            vals.append(float(np.min(drawdown)))
        else:
            vals.append(0.0)

        rets = _returns(arr)
        for win in [7, 14, 30]:
            if rets.size >= win:
                vals.append(float(np.mean(rets[-win:] > 0)))
            else:
                vals.append(0.0)

        if rets.size > 0:
            up_streak = 0
            for v in rets[::-1]:
                if v > 0:
                    up_streak += 1
                else:
                    break
            down_streak = 0
            for v in rets[::-1]:
                if v < 0:
                    down_streak += 1
                else:
                    break
            vals.append(min(up_streak / 30.0, 1.0))
            vals.append(min(down_streak / 30.0, 1.0))
        else:
            vals.extend([0.0, 0.0])

        # 가용성 마스크 (51~59)
        masks = [
            1.0 if arr.size >= 2 else 0.0,
            1.0 if arr.size >= 3 else 0.0,
            1.0 if arr.size >= 5 else 0.0,
            1.0 if arr.size >= 7 else 0.0,
            1.0 if arr.size >= 10 else 0.0,
            1.0 if arr.size >= 14 else 0.0,
            1.0 if arr.size >= 21 else 0.0,
            1.0 if arr.size >= 30 else 0.0,
            min(arr.size / 30.0, 1.0),
        ]

        features = vals + masks
        assert len(features) == 60
        return [_clip(v, -3.0, 3.0) for v in features]

    def calculate_exchange_features(self, exchange_data: dict, exchange_history: list[dict] | None = None) -> list:
        """환율 Feature (20차원)."""
        hist = exchange_history or []
        usd = float(exchange_data.get("USD/KRW", 0.0))
        jpy = float(exchange_data.get("JPY/KRW", 0.0))
        eur = float(exchange_data.get("EUR/KRW", 0.0))

        vals = [
            _safe_div(usd, 2000.0),  # 60
            _safe_div(jpy, 1500.0),  # 61
            _safe_div(eur, 2000.0),  # 62
            _safe_div(eur, usd),     # 63
        ]

        def series(key: str) -> np.ndarray:
            return np.asarray([float(x.get(key, 0.0)) for x in hist if float(x.get(key, 0.0)) > 0], dtype=np.float64)

        usd_s = series("USD/KRW")
        jpy_s = series("JPY/KRW")
        eur_s = series("EUR/KRW")

        for s in [usd_s, jpy_s, eur_s]:
            if s.size >= 2:
                vals.append(_safe_div(s[-1] - np.mean(s), np.mean(s)))  # dev from mean
            else:
                vals.append(0.0)
        for s in [usd_s, jpy_s, eur_s]:
            vals.append(_safe_div(s[-1] - s[-2], s[-2]) if s.size >= 2 else 0.0)  # 1d
        for s in [usd_s, jpy_s, eur_s]:
            vals.append(_safe_div(s[-1] - s[-7], s[-7]) if s.size >= 7 else 0.0)  # 7d
        for s in [usd_s, jpy_s, eur_s]:
            if s.size >= 7:
                vals.append(_safe_div(np.std(s[-7:]), np.mean(s[-7:])))
            else:
                vals.append(0.0)

        if usd_s.size >= 3 and eur_s.size >= 3:
            vals.append(float(np.corrcoef(usd_s[-min(30, usd_s.size):], eur_s[-min(30, eur_s.size):])[0, 1]))
        else:
            vals.append(0.0)
        if usd_s.size >= 3 and jpy_s.size >= 3:
            vals.append(float(np.corrcoef(usd_s[-min(30, usd_s.size):], jpy_s[-min(30, jpy_s.size):])[0, 1]))
        else:
            vals.append(0.0)
        if eur_s.size >= 3 and jpy_s.size >= 3:
            vals.append(float(np.corrcoef(eur_s[-min(30, eur_s.size):], jpy_s[-min(30, jpy_s.size):])[0, 1]))
        else:
            vals.append(0.0)

        coverage = min(min(usd_s.size, jpy_s.size, eur_s.size) / 30.0, 1.0)
        vals.append(coverage)

        if len(vals) < 20:
            vals.extend([0.0] * (20 - len(vals)))
        return [_clip(v, -3.0, 3.0) for v in vals[:20]]

    def calculate_news_features(
        self,
        news_stats: dict,
        articles: list[dict] | None = None,
        news_history: list[dict] | None = None,
    ) -> list:
        """뉴스 Feature (30차원)."""
        articles = articles or []
        hist = news_history or []

        sentiment_avg = float(news_stats.get("sentiment_avg", 0.0))
        total = float(news_stats.get("total", 0.0))
        pos = float(news_stats.get("positive_count", 0.0))
        neg = float(news_stats.get("negative_count", 0.0))
        neu = max(total - pos - neg, 0.0)

        vals = [
            _clip((sentiment_avg + 1.0) / 2.0, 0.0, 1.0),  # 80
            min(total / 100.0, 1.0),  # 81
            min(pos / 50.0, 1.0),     # 82
            min(neg / 50.0, 1.0),     # 83
            min(neu / 50.0, 1.0),
            _safe_div(pos, total),
            _safe_div(neg, total),
            min(abs(sentiment_avg), 1.0),
        ]

        titles = [str(a.get("title", "")) for a in articles if a.get("title")]
        if titles:
            lens = np.asarray([len(t) for t in titles], dtype=np.float64)
            vals.append(min(float(np.mean(lens)) / 200.0, 1.0))
            vals.append(min(float(np.std(lens)) / 100.0, 1.0))
        else:
            vals.extend([0.0, 0.0])

        text_blob = " ".join(t.lower() for t in titles)
        keywords = {
            "price": ["price", "가격", "drop", "하락", "상승", "인상"],
            "supply": ["stock", "재고", "shortage", "supply", "공급"],
            "release": ["release", "출시", "launch", "신제품"],
        }
        denom = max(len(titles), 1)
        vals.append(sum(any(k in t.lower() for k in keywords["price"]) for t in titles) / denom)
        vals.append(sum(any(k in t.lower() for k in keywords["supply"]) for t in titles) / denom)
        vals.append(sum(any(k in t.lower() for k in keywords["release"]) for t in titles) / denom)

        senti_series = np.asarray([float(x.get("sentiment_avg", 0.0)) for x in hist], dtype=np.float64)
        if senti_series.size >= 7:
            vals.append(_clip((float(np.mean(senti_series[-7:])) + 1.0) / 2.0, 0.0, 1.0))
            vals.append(_safe_div(senti_series[-1] - senti_series[-7], 2.0))
        else:
            vals.extend([0.0, 0.0])

        # 마스크/커버리지 구간 (15~29)
        vals.extend(
            [
                1.0 if total > 0 else 0.0,
                1.0 if titles else 0.0,
                1.0 if senti_series.size >= 2 else 0.0,
                1.0 if senti_series.size >= 7 else 0.0,
                1.0 if senti_series.size >= 30 else 0.0,
                min(senti_series.size / 30.0, 1.0),
                _safe_div(len(set(titles)), len(titles), default=0.0) if titles else 0.0,
                1.0 if "nvidia" in text_blob else 0.0,
                1.0 if "amd" in text_blob else 0.0,
                1.0 if "intel" in text_blob else 0.0,
                1.0 if "rtx" in text_blob else 0.0,
                1.0 if "radeon" in text_blob else 0.0,
                1.0 if "arc" in text_blob else 0.0,
                1.0 if "discount" in text_blob or "할인" in text_blob else 0.0,
                1.0 if "gaming" in text_blob else 0.0,
            ]
        )

        if len(vals) < 30:
            vals.extend([0.0] * (30 - len(vals)))
        return [_clip(v, -3.0, 3.0) for v in vals[:30]]

    def calculate_market_features(self, product_info: dict, today_products: list[dict] | None = None) -> list:
        """시장 공급 Feature (20차원)."""
        today_products = today_products or []
        current_price = float(product_info.get("lowest_price", 0.0))
        seller_count = float(product_info.get("seller_count", 0.0))
        stock_status = str(product_info.get("stock_status", "unknown"))
        model = str(product_info.get("chipset", ""))

        stock_status_map = {"in_stock": 1.0, "low_stock": 0.5, "out_of_stock": 0.0}
        vals = [
            min(seller_count / 50.0, 1.0),                # 110
            stock_status_map.get(stock_status, 0.0),      # 111
        ]

        prices = np.asarray([float(p.get("lowest_price", 0.0)) for p in today_products], dtype=np.float64)
        sellers = np.asarray([float(p.get("seller_count", 0.0)) for p in today_products], dtype=np.float64)
        stocks = [str(p.get("stock_status", "")) for p in today_products]

        if prices.size >= 2 and current_price > 0:
            vals.append(float(np.mean(prices <= current_price)))
        else:
            vals.append(0.0)
        if sellers.size >= 2:
            vals.append(float(np.mean(sellers <= seller_count)))
        else:
            vals.append(0.0)

        denom = max(len(stocks), 1)
        vals.append(stocks.count("in_stock") / denom)
        vals.append(stocks.count("low_stock") / denom)
        vals.append(stocks.count("out_of_stock") / denom)

        family = "other"
        if model.startswith("RTX"):
            family = "nvidia"
        elif model.startswith("RX"):
            family = "amd"
        elif model.startswith("Arc"):
            family = "intel"
        vals.extend(
            [
                1.0 if family == "nvidia" else 0.0,
                1.0 if family == "amd" else 0.0,
                1.0 if family == "intel" else 0.0,
            ]
        )

        same_family = [p for p in today_products if str(p.get("chipset", "")).startswith(model.split(" ")[0])]
        if len(same_family) >= 2 and current_price > 0:
            fam_prices = np.asarray([float(x.get("lowest_price", 0.0)) for x in same_family], dtype=np.float64)
            vals.append(float(np.mean(fam_prices <= current_price)))
        else:
            vals.append(0.0)

        vals.append(min(np.log1p(max(current_price, 0.0)) / 15.0, 1.0))
        if sellers.size >= 2:
            vals.append(_safe_div(seller_count - np.mean(sellers), np.std(sellers)))
        else:
            vals.append(0.0)
        vals.append(stock_status_map.get(stock_status, 0.0) * min(seller_count / 50.0, 1.0))
        vals.append(min(len(same_family) / max(len(today_products), 1), 1.0))

        vals.extend(
            [
                1.0 if today_products else 0.0,
                1.0 if len(same_family) >= 2 else 0.0,
                1.0 if sellers.size >= 2 else 0.0,
                1.0 if stock_status in {"in_stock", "low_stock", "out_of_stock"} else 0.0,
            ]
        )

        if len(vals) < 20:
            vals.extend([0.0] * (20 - len(vals)))
        return [_clip(v, -3.0, 3.0) for v in vals[:20]]

    def calculate_time_features(self, date_str: str) -> list:
        """시간 Feature (20차원) - 전부 날짜 기반 실계산."""
        date = datetime.strptime(date_str, "%Y-%m-%d")
        day_of_week = date.weekday()  # 0~6
        month = date.month
        day_of_month = date.day
        day_of_year = date.timetuple().tm_yday
        iso_week = date.isocalendar().week
        quarter = (month - 1) // 3 + 1

        features = [
            day_of_week / 7.0,                # 130
            month / 12.0,                     # 131
            1.0 if month == 12 else 0.0,      # 132
            day_of_month / 31.0,
            day_of_year / 366.0,
            iso_week / 53.0,
            quarter / 4.0,
            1.0 if day_of_week >= 5 else 0.0,
            1.0 if day_of_month <= 3 else 0.0,
            1.0 if day_of_month >= 28 else 0.0,
            np.sin(2 * np.pi * day_of_week / 7.0),
            np.cos(2 * np.pi * day_of_week / 7.0),
            np.sin(2 * np.pi * month / 12.0),
            np.cos(2 * np.pi * month / 12.0),
            np.sin(2 * np.pi * day_of_year / 366.0),
            np.cos(2 * np.pi * day_of_year / 366.0),
            1.0 if quarter == 1 else 0.0,
            1.0 if quarter == 2 else 0.0,
            1.0 if quarter == 3 else 0.0,
            1.0 if quarter == 4 else 0.0,
        ]
        return [_clip(float(v), -1.0, 1.0) for v in features]

    def calculate_technical_indicators(self, historical_prices: list) -> list:
        """기술적 지표 (106차원) - 값 53 + 결측 마스크 53."""
        arr = np.asarray(historical_prices, dtype=np.float64)
        values: list[float] = []
        masks: list[float] = []

        # 기존 의미 보존: 150 RSI, 151 MACD, 152 Momentum
        rsi14, m_rsi14 = _rsi(arr, 14)
        values.append(rsi14)
        masks.append(m_rsi14)

        if arr.size >= 26:
            ema12, _ = _ema(arr, 12)
            ema26, _ = _ema(arr, 26)
            values.append(_safe_div(ema12 - ema26, ema26))  # MACD
            masks.append(1.0)
        else:
            values.append(0.0)
            masks.append(0.0)

        if arr.size >= 10:
            values.append(_safe_div(arr[-1] - arr[-10], arr[-10]))  # Momentum(10)
            masks.append(1.0)
        else:
            values.append(0.0)
            masks.append(0.0)

        # 3 + 50 = 53 values
        for win in [3, 5, 7, 10, 14, 21, 30]:
            if arr.size >= win:
                sma = float(np.mean(arr[-win:]))
                values.append(_safe_div(arr[-1] - sma, sma))
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for span in [3, 5, 7, 10, 14, 21, 30]:
            ema, m = _ema(arr, span)
            values.append(_safe_div(arr[-1] - ema, ema) if m else 0.0)
            masks.append(m)

        for lag in [3, 5, 7, 10, 14, 21, 30]:
            if arr.size >= lag:
                values.append(_safe_div(arr[-1] - arr[-lag], arr[-lag]))
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        rets = _returns(arr)
        for win in [3, 5, 7, 10, 14, 21, 30]:
            if rets.size >= win:
                values.append(float(np.std(rets[-win:])))
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for win in [7, 14, 21, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                mean = float(np.mean(seg))
                std = float(np.std(seg))
                values.append(_safe_div(arr[-1] - mean, std))           # BB zscore
                values.append(_safe_div(4.0 * std, mean))               # BB width
                masks.extend([1.0, 1.0])
            else:
                values.extend([0.0, 0.0])
                masks.extend([0.0, 0.0])

        for period in [6, 21]:
            rsi, m = _rsi(arr, period)
            values.append(rsi)
            masks.append(m)

        for win in [7, 14, 21]:
            if arr.size >= win:
                seg = arr[-win:]
                values.append(_safe_div(arr[-1] - np.min(seg), np.max(seg) - np.min(seg)))  # stochastic K
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for win in [10, 20, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                tp = float(np.mean(seg))
                mad = float(np.mean(np.abs(seg - tp)))
                values.append(_safe_div(arr[-1] - tp, 0.015 * mad))  # CCI-like
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for win in [7, 14, 21]:
            if arr.size >= win:
                seg = arr[-win:]
                tr = np.abs(np.diff(seg))
                values.append(_safe_div(float(np.mean(tr)) if tr.size > 0 else 0.0, float(np.mean(seg))))
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        for win in [7, 14, 30]:
            if arr.size >= win:
                seg = arr[-win:]
                peak = np.maximum.accumulate(seg)
                dd = (seg - peak) / np.maximum(peak, 1.0)
                values.append(float(np.min(dd)))
                masks.append(1.0)
            else:
                values.append(0.0)
                masks.append(0.0)

        assert len(values) == 53
        assert len(masks) == 53
        features = values + masks
        assert len(features) == 106
        return [_clip(v, -3.0, 3.0) for v in features]

    def generate_features(self, gpu_model: str, date_str: str, context: dict | None = None) -> np.ndarray:
        """특정 GPU 모델의 256차원 Feature 생성."""
        ctx = context if context is not None else self._build_feature_context(date_str)
        if not ctx:
            return np.zeros(256, dtype=np.float32)

        historical_prices = ctx["historical_data"].get(gpu_model, [])
        today_products = ctx["today_products"]
        product_info = ctx["product_by_model"].get(gpu_model)
        if product_info is None:
            logger.warning(f"제품 정보 없음: {gpu_model}")
            return np.zeros(256, dtype=np.float32)

        current_price = float(product_info.get("lowest_price", 0.0))

        features: list[float] = []
        features.extend(self.calculate_price_features(current_price, historical_prices))  # 60
        features.extend(self.calculate_exchange_features(ctx["exchange_data"], ctx["exchange_history"]))  # 20
        features.extend(self.calculate_news_features(ctx["news_stats"], ctx["today_articles"], ctx["news_history"]))  # 30
        features.extend(self.calculate_market_features(product_info, today_products))  # 20
        features.extend(self.calculate_time_features(date_str))  # 20
        features.extend(self.calculate_technical_indicators(historical_prices))  # 106

        if len(features) != 256:
            raise RuntimeError(f"Feature dimension mismatch: expected 256, got {len(features)}")

        return np.asarray(features, dtype=np.float32)

    def process_all(self, date_str: str | None = None) -> str | None:
        """모든 GPU 모델 Feature 생성."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        logger.info("=" * 80)
        logger.info(f"Feature Engineering 시작 - {date_str}")
        logger.info("=" * 80)

        danawa_file = self.raw_data_dir / "danawa" / f"{date_str}.json"
        if not danawa_file.exists():
            logger.error(f"데이터 파일 없음: {danawa_file}")
            return None

        output_dir = self.processed_dir / "dataset"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"training_data_{date_str}.json"
        context = self._build_feature_context(date_str)
        if context is None:
            return None

        gpu_models = list(context["product_by_model"].keys())
        sample_count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("[\n")
            for idx, gpu_model in enumerate(gpu_models):
                logger.info(f"처리 중: {gpu_model}")
                feature_vector = self.generate_features(gpu_model, date_str, context=context)
                row = {
                    "date": date_str,
                    "gpu_model": gpu_model,
                    "state_vector": feature_vector.tolist(),
                }
                if idx > 0:
                    f.write(",\n")
                json.dump(row, f, ensure_ascii=False)
                sample_count += 1
            f.write("\n]\n")

        logger.info(f"\n✓ Feature 생성 완료: {sample_count}개 샘플")
        logger.info(f"✓ 저장 위치: {output_file}")
        logger.info("✓ Feature 차원: 256")
        return str(output_file)


if __name__ == "__main__":
    engineer = FeatureEngineer()
    engineer.process_all()
