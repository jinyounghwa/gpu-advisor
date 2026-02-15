"""
Custom Tokenizer for Financial News
30MB 영문 뉴스에서 핵심 키워드(Vocabulary) 추출
"""

import re
import json
from typing import List, Dict, Set, Tuple
from collections import Counter
from pathlib import Path


class NewsTokenizer:
    """
    금융 뉴스 전용 커스텀 토크나이저

    특징:
    1. 핵심 키워드 중심 Vocabulary 구축
    2. 소문자 정규화 + 불용어 제거
    3. 숫자/특수문자 처리
    4. 도메인별 특수 토큰 (NVIDIA, Shortage, Fed, Rate 등)
    """

    def __init__(
        self,
        vocab_size: int = 5000,
        min_freq: int = 2,
        domain_keywords: List[str] = None,
    ):
        """
        Args:
            vocab_size: 최대 어휘 크기
            min_freq: 최소 등장 빈도
            domain_keywords: 금융 도메인 특수 키워드
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq

        # 금융 도메인 특수 키워드 (항상 포함)
        self.domain_keywords = domain_keywords or [
            "nvidia",
            "gpu",
            "graphics",
            "shortage",
            "supply",
            "demand",
            "fed",
            "rate",
            "interest",
            "inflation",
            "cpi",
            "price",
            "market",
            "buy",
            "sell",
            "trade",
            "stock",
            "crypto",
            "bitcoin",
            "ethereum",
            "amd",
            "radeon",
            "tariff",
            "semiconductor",
            "chip",
            "ai",
            "ml",
        ]

        # 특수 토큰
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"
        self.START_TOKEN = "<START>"
        self.END_TOKEN = "<END>"

        # 불용어
        self.stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "from",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "as",
            "of",
            "that",
            "this",
            "it",
            "its",
            "they",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
            "now",
            "but",
        }

        # Vocabulary: word → idx
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()

        # Vocabulary 초기화
        self._init_vocab()

    def _init_vocab(self):
        """특수 토큰 및 도메인 키워드로 Vocabulary 초기화"""
        # 특수 토큰 먼저 추가
        special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.START_TOKEN,
            self.END_TOKEN,
        ]

        for token in special_tokens:
            idx = len(self.word2idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

        # 도메인 키워드 추가 (항상 포함)
        for keyword in self.domain_keywords:
            keyword_lower = keyword.lower()
            if keyword_lower not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[keyword_lower] = idx
                self.idx2word[idx] = keyword_lower
                self.word_freq[keyword_lower] = self.min_freq  # 최소 빈도로 설정

    def _tokenize_text(self, text: str) -> List[str]:
        """
        텍스트 토크나이징

        Args:
            text: 원본 텍스트

        Returns:
            토큰 리스트
        """
        # 소문자 정규화
        text = text.lower()

        # URL, 이메일 제거
        text = re.sub(r"http\S+|www.\S+|https\S+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\S*@\S*\s?", "", text, flags=re.MULTILINE)

        # 숫자: <NUM> 토큰으로 대체
        text = re.sub(r"\d+", "<NUM>", text)

        # 특수문자 제거 (알파벳, 숫자, 공백, < > 유지)
        text = re.sub(r"[^a-z0-9\s<>]", " ", text)

        # 여러 공백을 하나로
        text = re.sub(r"\s+", " ", text).strip()

        # 토큰화
        tokens = text.split()

        # 불용어 제거
        tokens = [t for t in tokens if t not in self.stop_words]

        # 너무 짧은 토큰 제거
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def build_vocab(self, articles: List[str]) -> None:
        """
        뉴스 기사들로 Vocabulary 구축

        Args:
            articles: 뉴스 기사 텍스트 리스트
        """
        # 모든 기사 토크나이징
        all_tokens = []
        for article in articles:
            tokens = self._tokenize_text(article)
            all_tokens.extend(tokens)

        # 빈도 계산
        self.word_freq = Counter(all_tokens)

        # 도메인 키워드 빈도 업데이트
        for keyword in self.domain_keywords:
            self.word_freq[keyword.lower()] = max(
                self.word_freq.get(keyword.lower(), 0), self.min_freq
            )

        # 빈도순 정렬
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Vocabulary 채우기
        # 이미 추가된 특수 토큰과 도메인 키워드는 건너뜀
        current_size = len(self.word2idx)

        for word, freq in sorted_words:
            if word in self.word2idx:
                continue  # 이미 추가됨

            if freq >= self.min_freq and current_size < self.vocab_size:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                current_size += 1

            if current_size >= self.vocab_size:
                break

        print(f"Vocabulary built: {len(self.word2idx)} words")

    def encode(self, text: str, max_length: int = 128) -> List[int]:
        """
        텍스트를 정수 인덱스로 인코딩

        Args:
            text: 원본 텍스트
            max_length: 최대 시퀀스 길이

        Returns:
            정수 인덱스 리스트
        """
        tokens = self._tokenize_text(text)

        # 토큰을 정수 인덱스로 변환
        indices = [self.START_TOKEN]
        for token in tokens[: max_length - 2]:  # START, END 토큰 공간 확보
            idx = self.word2idx.get(token, self.word2idx[self.UNK_TOKEN])
            indices.append(idx)
        indices.append(self.END_TOKEN)

        # 패딩
        if len(indices) < max_length:
            indices += [self.word2idx[self.PAD_TOKEN]] * (max_length - len(indices))
        else:
            indices = indices[:max_length]

        return indices

    def decode(self, indices: List[int]) -> str:
        """
        정수 인덱스를 텍스트로 디코딩

        Args:
            indices: 정수 인덱스 리스트

        Returns:
            디코딩된 텍스트
        """
        tokens = []
        for idx in indices:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]:
                continue
            tokens.append(word)

        return " ".join(tokens)

    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return len(self.word2idx)

    def save_vocab(self, filepath: str) -> None:
        """
        Vocabulary 저장

        Args:
            filepath: 저장 경로
        """
        vocab_data = {
            "word2idx": self.word2idx,
            "idx2word": self.idx2word,
            "word_freq": dict(self.word_freq),
            "vocab_size": self.vocab_size,
            "min_freq": self.min_freq,
            "domain_keywords": self.domain_keywords,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)

        print(f"Vocabulary saved to {filepath}")

    def load_vocab(self, filepath: str) -> None:
        """
        Vocabulary 로드

        Args:
            filepath: 로드 경로
        """
        with open(filepath, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        self.word2idx = {k: int(v) for k, v in vocab_data["word2idx"].items()}
        self.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
        self.word_freq = Counter(vocab_data["word_freq"])
        self.vocab_size = vocab_data["vocab_size"]
        self.min_freq = vocab_data["min_freq"]
        self.domain_keywords = vocab_data["domain_keywords"]

        print(f"Vocabulary loaded from {filepath}: {len(self.word2idx)} words")


def load_news_articles(news_dir: Path) -> List[str]:
    """
    뉴스 디렉토리에서 모든 기사 로드

    Args:
        news_dir: 뉴스 데이터 디렉토리

    Returns:
        기사 텍스트 리스트
    """
    articles = []

    news_files = list(news_dir.glob("*.json"))
    print(f"Found {len(news_files)} news files")

    for news_file in news_files:
        with open(news_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 기사들 추출
        for article in data.get("articles", []):
            if "title" in article and "content" in article:
                text = f"{article['title']} {article['content']}"
                articles.append(text)

    print(f"Loaded {len(articles)} articles")

    return articles


if __name__ == "__main__":
    # 테스트
    data_dir = Path("/Users/younghwa.jin/Documents/gpu-advisor/data")
    news_dir = data_dir / "raw" / "news"

    # 뉴스 기사 로드
    articles = load_news_articles(news_dir)

    # 실제 뉴스가 없으면 더미 데이터 사용
    if len(articles) == 0:
        print("No articles found, using dummy data...")
        articles = [
            "NVIDIA GPU shortage continues as demand surges for AI training",
            "Fed rate decision impacts crypto market volatility",
            "AMD Radeon prices drop amid competitive pressure",
            "Semiconductor supply chain faces tariff challenges",
            "Bitcoin and Ethereum show bullish sentiment",
        ] * 100  # 더미 데이터 확장

    # 토크나이저 생성
    tokenizer = NewsTokenizer(vocab_size=5000, min_freq=2)

    # Vocabulary 구축
    tokenizer.build_vocab(articles)

    # 테스트 인코딩
    test_text = "NVIDIA GPU prices drop amid competitive pressure from AMD"
    encoded = tokenizer.encode(test_text, max_length=32)
    decoded = tokenizer.decode(encoded)

    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")

    # Vocabulary 저장
    vocab_path = data_dir / "processed" / "tokenizer_vocab.json"
    vocab_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save_vocab(str(vocab_path))
