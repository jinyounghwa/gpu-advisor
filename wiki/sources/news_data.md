# 뉴스 감성 분석 데이터

> 마지막 업데이트: 2026-04-11

## 개요

Google News RSS에서 GPU 관련 뉴스를 수집하고 감성 분석을 수행합니다.

## 수집 키워드

- NVIDIA GPU price
- AMD Radeon GPU
- Intel Arc GPU
- graphics card price
- GPU shortage
- RTX 5090, RTX 5080
- GPU stock

## 감성 분석

| 감성 | 기준 |
|------|------|
| positive | score > 0 |
| neutral | score = 0 |
| negative | score < 0 |

## 최근 통계 (2026-04-11)

- 총 기사: 37개
- 평균 감성: -0.03 (중립)
- 긍정: 5개, 부정: 6개, 중립: 26개

## 저장 위치

`data/raw/news/YYYY-MM-DD.json`

## 크롤러

`crawlers/news_crawler.py` — Google News RSS + 감성 분석

## 관련 페이지

- [[analysis/market_sentiment]] — 감성 분석 종합
- [[concepts/feature_engineering]] — 뉴스 → Feature 변환

---

[[index|← 인덱스로 돌아가기]]
