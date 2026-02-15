"""
Numerical Data Scaler
가격, 환율 등 5가지 수치 데이터를 0~1로 정규화
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import json
from pathlib import Path


class NumericalScaler:
    """
    수치 데이터 정규화 클래스

    특징:
    1. MinMaxScaler: 0~1 범위로 정규화
    2. StandardScaler: 표준화 (평균 0, 표준편차 1)
    3. 스케일러 저장/로드 기능
    4. 배치 처리 지원
    """

    def __init__(
        self,
        method: str = "minmax",
        feature_names: List[str] = None,
    ):
        """
        Args:
            method: 정규화 방법 ('minmax' or 'standard')
            feature_names: 특성 이름 리스트
        """
        self.method = method
        self.feature_names = feature_names or [
            "price",  # 가격
            "price_change_24h",  # 24h 변동률
            "volume_24h",  # 24h 거래량
            "market_cap",  # 시가총액
            "volatility",  # 변동성
        ]

        # 스케일러 초기화
        if method == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        elif method == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {method}")

        # 스케일러가 학습되었는지 확인
        self.is_fitted = False

        # 통계 정보 저장
        self.stats: Dict[str, Dict] = {}

    def fit(self, data: np.ndarray) -> None:
        """
        스케일러 학습

        Args:
            data: 학습 데이터 (num_samples, num_features)
        """
        self.scaler.fit(data)
        self.is_fitted = True

        # 통계 정보 저장
        for i, name in enumerate(self.feature_names):
            if self.method == "minmax":
                self.stats[name] = {
                    "min": float(self.scaler.data_min_[i]),
                    "max": float(self.scaler.data_max_[i]),
                }
            else:
                self.stats[name] = {
                    "mean": float(self.scaler.mean_[i]),
                    "std": float(self.scaler.scale_[i]),
                }

        print(f"Scaler fitted on data shape: {data.shape}")

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        데이터 변환

        Args:
            data: 변환할 데이터 (num_samples, num_features)

        Returns:
            정규화된 데이터
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        return self.scaler.transform(data)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        스케일러 학습 후 변환

        Args:
            data: 학습 및 변환할 데이터

        Returns:
            정규화된 데이터
        """
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        원본 스케일로 역변환

        Args:
            data: 정규화된 데이터

        Returns:
            원본 스케일 데이터
        """
        if not self.is_fitted:
            raise RuntimeError("Scaler not fitted. Call fit() first.")

        return self.scaler.inverse_transform(data)

    def save_scaler(self, filepath: str) -> None:
        """
        스케일러 저장

        Args:
            filepath: 저장 경로
        """
        scaler_data = {
            "method": self.method,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "stats": self.stats,
            "scale_": self.scaler.scale_.tolist()
            if hasattr(self.scaler, "scale_")
            else None,
            "min_": self.scaler.data_min_.tolist()
            if hasattr(self.scaler, "data_min_")
            else None,
            "max_": self.scaler.data_max_.tolist()
            if hasattr(self.scaler, "data_max_")
            else None,
            "mean_": self.scaler.mean_.tolist()
            if hasattr(self.scaler, "mean_")
            else None,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(scaler_data, f, ensure_ascii=False, indent=2)

        print(f"Scaler saved to {filepath}")

    def load_scaler(self, filepath: str) -> None:
        """
        스케일러 로드

        Args:
            filepath: 로드 경로
        """
        with open(filepath, "r", encoding="utf-8") as f:
            scaler_data = json.load(f)

        self.method = scaler_data["method"]
        self.feature_names = scaler_data["feature_names"]
        self.is_fitted = scaler_data["is_fitted"]
        self.stats = scaler_data["stats"]

        if self.is_fitted:
            if self.method == "minmax":
                self.scaler.min_ = np.array(scaler_data["min_"])
                self.scaler.max_ = np.array(scaler_data["max_"])
                self.scaler.scale_ = (self.scaler.max_ - self.scaler.min_) / (
                    self.scaler.data_range_
                    if hasattr(self.scaler, "data_range_")
                    else 1
                )
            elif self.method == "standard":
                self.scaler.mean_ = np.array(scaler_data["mean_"])
                self.scaler.scale_ = np.array(scaler_data["scale_"])

        print(f"Scaler loaded from {filepath}")


class MultiFeatureScaler:
    """
    여러 특성에 대해 개별적으로 스케일링
    """

    def __init__(self, feature_configs: List[Dict[str, str]] = None):
        """
        Args:
            feature_configs: 각 특성의 설정 리스트
                [{"name": "price", "method": "minmax"}, ...]
        """
        self.feature_configs = feature_configs or []
        self.scalers: Dict[str, NumericalScaler] = {}

    def add_feature(self, name: str, method: str = "minmax") -> None:
        """
        특성 추가

        Args:
            name: 특성 이름
            method: 스케일링 방법
        """
        self.feature_configs.append({"name": name, "method": method})
        self.scalers[name] = NumericalScaler(method=method, feature_names=[name])

    def fit(self, data: Dict[str, np.ndarray]) -> None:
        """
        모든 스케일러 학습

        Args:
            data: 특성별 데이터 딕셔너리
        """
        for config in self.feature_configs:
            name = config["name"]
            scaler = self.scalers[name]
            scaler.fit(data[name].reshape(-1, 1))

    def transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        모든 데이터 변환

        Args:
            data: 특성별 데이터 딕셔너리

        Returns:
            정규화된 데이터 딕셔너리
        """
        transformed = {}
        for name, values in data.items():
            if name in self.scalers:
                transformed[name] = (
                    self.scalers[name].transform(values.reshape(-1, 1)).flatten()
                )
            else:
                transformed[name] = values

        return transformed

    def fit_transform(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        스케일러 학습 후 변환

        Args:
            data: 특성별 데이터 딕셔너리

        Returns:
            정규화된 데이터 딕셔너리
        """
        for config in self.feature_configs:
            name = config["name"]
            if name in data:
                scaler = self.scalers[name]
                data[name] = scaler.fit_transform(data[name].reshape(-1, 1)).flatten()

        return data


def create_dummy_training_data(num_samples: int = 1000) -> np.ndarray:
    """
    더미 학습 데이터 생성

    Args:
        num_samples: 샘플 수

    Returns:
        생성된 데이터 (num_samples, 5)
    """
    np.random.seed(42)

    data = np.zeros((num_samples, 5))

    # 가격: 100 ~ 10000
    data[:, 0] = np.random.uniform(100, 10000, num_samples)

    # 24h 변동률: -10% ~ +10%
    data[:, 1] = np.random.uniform(-0.1, 0.1, num_samples)

    # 24h 거래량: 0 ~ 1000000
    data[:, 2] = np.random.uniform(0, 1000000, num_samples)

    # 시가총액: 1000000 ~ 10000000000
    data[:, 3] = np.random.uniform(1e6, 1e10, num_samples)

    # 변동성: 0 ~ 1
    data[:, 4] = np.random.uniform(0, 1, num_samples)

    return data


if __name__ == "__main__":
    # 테스트
    print("Numerical Scaler Test")

    # 더미 데이터 생성
    dummy_data = create_dummy_training_data(1000)
    print(f"\nDummy data shape: {dummy_data.shape}")
    print(f"First sample (original): {dummy_data[0]}")

    # 스케일러 생성
    scaler = NumericalScaler(method="minmax")

    # 스케일러 학습
    scaler.fit(dummy_data)

    # 변환
    scaled_data = scaler.transform(dummy_data)
    print(f"\nFirst sample (scaled): {scaled_data[0]}")

    # 역변환
    inverse_data = scaler.inverse_transform(scaled_data)
    print(f"First sample (inverse): {inverse_data[0]}")

    # 스케일러 저장
    data_dir = Path("/Users/younghwa.jin/Documents/gpu-advisor/data")
    scaler_path = data_dir / "processed" / "scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler.save_scaler(str(scaler_path))

    # 로드 테스트
    new_scaler = NumericalScaler(method="minmax")
    new_scaler.load_scaler(str(scaler_path))

    # 로드된 스케일러로 변환
    new_scaled = new_scaler.transform(dummy_data)
    print(f"\nFirst sample (loaded scaler): {new_scaled[0]}")
    print(f"Matches original: {np.allclose(scaled_data, new_scaled)}")
