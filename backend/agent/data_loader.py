"""
Centralized data loader for Agent datasets.
Handles loading processed state vectors and raw prices from the filesystem.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np


class AgentDataLoader:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.dataset_dir = project_root / "data" / "processed" / "dataset"
        self.raw_danawa_dir = project_root / "data" / "raw" / "danawa"

    def get_latest_dataset_path(self) -> Path:
        dated_files = sorted(self.dataset_dir.glob("training_data_*.json"))
        if dated_files:
            return dated_files[-1]
        fallback = self.dataset_dir / "training_data.json"
        if fallback.exists():
            return fallback
        raise FileNotFoundError("No processed training dataset file found")

    def load_latest_dataset(self) -> List[Dict[str, Any]]:
        path = self.get_latest_dataset_path()
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_processed_by_date(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Returns mapping: date -> gpu_model -> state_vector"""
        files = sorted(self.dataset_dir.glob("training_data_*.json"))
        by_date: Dict[str, Dict[str, np.ndarray]] = {}
        for f in files:
            date = f.stem.replace("training_data_", "")
            with open(f, "r", encoding="utf-8") as fp:
                rows = json.load(fp)
            by_date[date] = {
                row["gpu_model"]: np.asarray(row["state_vector"], dtype=np.float32)
                for row in rows
            }
        return by_date

    def load_prices_by_date(self) -> Dict[str, Dict[str, float]]:
        """Returns mapping: date -> chipset -> lowest_price"""
        prices: Dict[str, Dict[str, float]] = {}
        for f in sorted(self.raw_danawa_dir.glob("*.json")):
            date = f.stem
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                prices[date] = {
                    p["chipset"]: float(p["lowest_price"])
                    for p in data.get("products", [])
                }
            except Exception:
                continue
        return prices
