"""Sentiment analyzer with optional model backend + safe fallback."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List


class _RuleBasedScorer:
    POSITIVE_TERMS = {
        "drop", "discount", "stabilize", "stable", "improve", "recovery", "recover",
        "boost", "upside", "strong", "cheaper", "availability", "in stock", "surplus",
    }
    NEGATIVE_TERMS = {
        "surge", "spike", "shortage", "delay", "ban", "risk", "uncertainty",
        "inflation", "expensive", "out of stock", "tight", "volatile", "volatility",
    }

    def score(self, text: str) -> float:
        lowered = (text or "").lower()
        if not lowered.strip():
            return 0.0
        pos = sum(1 for term in self.POSITIVE_TERMS if term in lowered)
        neg = sum(1 for term in self.NEGATIVE_TERMS if term in lowered)
        raw = pos - neg
        return math.tanh(raw / 3.0)


class _TransformerScorer:
    """Optional model-based scorer using HuggingFace pipeline."""

    def __init__(self, model_name: str):
        from transformers import pipeline  # type: ignore

        self.classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            truncation=True,
        )

    def score(self, text: str) -> float:
        if not text.strip():
            return 0.0
        out = self.classifier(text[:512])[0]
        label = str(out.get("label", "")).upper()
        confidence = float(out.get("score", 0.5))
        signed = confidence if ("POS" in label or label.endswith("5")) else -confidence
        return max(-1.0, min(1.0, signed))


class NewsSentimentAnalyzer:
    """Model-first analyzer with fallback to rule-based scoring."""

    def __init__(self):
        self.backend_requested = os.getenv("GPU_ADVISOR_SENTIMENT_BACKEND", "auto").strip().lower()
        model_name = os.getenv(
            "GPU_ADVISOR_SENTIMENT_MODEL",
            "distilbert-base-uncased-finetuned-sst-2-english",
        ).strip()
        self.backend_used = "rule"
        self.fallback_reason = None
        self.scorer = _RuleBasedScorer()

        should_try_model = self.backend_requested in {"auto", "transformers"}
        if should_try_model:
            try:
                self.scorer = _TransformerScorer(model_name=model_name)
                self.backend_used = "transformers"
            except Exception as e:
                self.backend_used = "rule"
                self.fallback_reason = f"transformers_unavailable:{e.__class__.__name__}"

    def score_text(self, text: str) -> float:
        return self.scorer.score(text)

    def aggregate(self, articles: Iterable[Dict]) -> Dict[str, float | int]:
        scores: List[float] = []
        for article in articles:
            title = str(article.get("title", ""))
            summary = str(article.get("summary", ""))
            text = f"{title} {summary}".strip()
            model_score = self.score_text(text)

            # Crawl output may already include sentiment_score; blend if present.
            source_score = article.get("sentiment_score")
            if isinstance(source_score, (int, float)):
                blended = 0.6 * float(source_score) + 0.4 * model_score
            else:
                blended = model_score
            scores.append(max(-1.0, min(1.0, blended)))

        if not scores:
            return {
                "total": 0,
                "sentiment_avg": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
            }

        positive_count = sum(1 for score in scores if score > 0.1)
        negative_count = sum(1 for score in scores if score < -0.1)
        neutral_count = len(scores) - positive_count - negative_count
        return {
            "total": len(scores),
            "sentiment_avg": float(sum(scores) / len(scores)),
            "positive_count": int(positive_count),
            "negative_count": int(negative_count),
            "neutral_count": int(neutral_count),
            "backend_used": self.backend_used,
            "fallback_reason": self.fallback_reason,
        }

    def analyze_news_file(self, news_file: Path) -> Dict[str, float | int]:
        with open(news_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        articles = payload.get("articles", [])
        result = self.aggregate(articles)
        result["date"] = payload.get("date")
        result["source"] = payload.get("source", "unknown")
        result["file"] = str(news_file)
        result["backend_requested"] = self.backend_requested
        return result
