# 환율 데이터

> 마지막 업데이트: 2026-04-11

## 개요

GPU 가격에 영향을 미치는 주요 환율을 매일 수집합니다.

## 수집 환율

| 통화쌍 | 최근 값 (2026-04-11) | 용도 |
|--------|---------------------|------|
| USD/KRW | 1,476.98 | NVIDIA GPU 가격과 직접 연관 |
| JPY/KRW | 929.06 | AMD GPU 가격 참고 |
| EUR/KRW | 1,726.61 | 유럽 수입 제품 참고 |

## 소스

open.er-api.com (무료 환율 API)

## 저장 위치

`data/raw/exchange/YYYY-MM-DD.json`

## 크롤러

`crawlers/exchange_rate_crawler.py`

## 관련 페이지

- [[concepts/feature_engineering]] — 환율 → Feature 변환
- [[concepts/market_indicators]] — 기술적 지표

---

[[index|← 인덱스로 돌아가기]]
