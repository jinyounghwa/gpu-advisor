"""Tests for feature engineering pipeline."""

import pytest
import numpy as np
from crawlers.feature_engineer import FeatureEngineer


class TestPriceFeatures:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        prices = [500000 + i * 1000 for i in range(30)]
        features = self.eng.calculate_price_features(510000, prices)
        assert len(features) == 60

    def test_no_history(self):
        features = self.eng.calculate_price_features(500000, [])
        assert len(features) == 60
        # First element should be normalized price
        assert features[0] == pytest.approx(500000 / 10000000)

    def test_short_history(self):
        features = self.eng.calculate_price_features(500000, [490000, 500000])
        assert len(features) == 60

    def test_values_bounded(self):
        prices = [500000 + i * 1000 for i in range(30)]
        features = self.eng.calculate_price_features(510000, prices)
        # Normalized prices should be small positive numbers
        assert features[0] > 0
        assert features[0] < 1


class TestExchangeFeatures:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        data = {"USD/KRW": 1400, "JPY/KRW": 900, "EUR/KRW": 1500}
        features = self.eng.calculate_exchange_features(data)
        assert len(features) == 20

    def test_empty_data(self):
        features = self.eng.calculate_exchange_features({})
        assert len(features) == 20


class TestNewsFeatures:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        stats = {"sentiment_avg": 0.5, "total": 20, "positive_count": 15, "negative_count": 5}
        features = self.eng.calculate_news_features(stats)
        assert len(features) == 30

    def test_empty_stats(self):
        features = self.eng.calculate_news_features({})
        assert len(features) == 30


class TestMarketFeatures:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        info = {"seller_count": 25, "stock_status": "in_stock"}
        features = self.eng.calculate_market_features(info)
        assert len(features) == 20

    def test_stock_status_mapping(self):
        info_in = {"seller_count": 10, "stock_status": "in_stock"}
        info_out = {"seller_count": 10, "stock_status": "out_of_stock"}
        f_in = self.eng.calculate_market_features(info_in)
        f_out = self.eng.calculate_market_features(info_out)
        assert f_in[1] == 1.0
        assert f_out[1] == 0.0


class TestTimeFeatures:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        features = self.eng.calculate_time_features("2026-02-21")
        assert len(features) == 20

    def test_year_end(self):
        features_dec = self.eng.calculate_time_features("2026-12-15")
        features_jun = self.eng.calculate_time_features("2026-06-15")
        assert features_dec[2] == 1.0
        assert features_jun[2] == 0.0


class TestTechnicalIndicators:
    def setup_method(self):
        self.eng = FeatureEngineer()

    def test_output_length(self):
        prices = list(range(500000, 530000, 1000))
        features = self.eng.calculate_technical_indicators(prices)
        assert len(features) == 106

    def test_short_history(self):
        features = self.eng.calculate_technical_indicators([500000])
        assert len(features) == 106
        # RSI unavailable -> 0 + mask(결측 처리)
        assert features[0] == 0.0

    def test_rsi_bounded(self):
        prices = list(range(500000, 530000, 1000))
        features = self.eng.calculate_technical_indicators(prices)
        assert 0 <= features[0] <= 1


class TestFullFeatureVector:
    def test_total_dimensions(self):
        eng = FeatureEngineer()
        price_f = eng.calculate_price_features(500000, [490000] * 30)
        exchange_f = eng.calculate_exchange_features({"USD/KRW": 1400})
        news_f = eng.calculate_news_features({})
        market_f = eng.calculate_market_features({"seller_count": 10})
        time_f = eng.calculate_time_features("2026-02-21")
        tech_f = eng.calculate_technical_indicators([490000] * 30)

        total = price_f + exchange_f + news_f + market_f + time_f + tech_f
        assert len(total) == 256
