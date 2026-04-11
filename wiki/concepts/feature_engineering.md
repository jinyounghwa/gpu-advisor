# Feature Engineering — 256차원 상태 벡터

> 마지막 업데이트: 2026-04-11

## 개요

Raw JSON 데이터를 256차원 상태 벡터로 변환하는 파이프라인. AlphaZero 신경망의 입력으로 사용됩니다.

## 특징 분류

| 카테고리 | 차원 수 | 설명 |
|----------|---------|------|
| Price Features | 60 | 가격 정규화, 변동성, 이동평균 |
| Exchange Features | 20 | USD/KRW, JPY/KRW, EUR/KRW 트렌드 |
| News Features | 30 | 감성 분석, 키워드 빈도 |
| Market Features | 20 | 재고 상태, 판매자 수 |
| Time Features | 20 | 요일, 월, 분기 |
| Technical Indicators | 106 | RSI, MACD, Bollinger Bands |

## 데이터 소스

- [[sources/danawa_data]] — GPU 가격 정보
- [[sources/exchange_data]] — 환율 정보
- [[sources/news_data]] — 뉴스 감성 분석

## 파이프라인

```
data/raw/*.json → feature_engineer.py → data/processed/dataset/training_data_YYYY-MM-DD.json
```

## 관련 페이지

- [[concepts/alphazero]] — 신경망 아키텍처
- [[concepts/market_indicators]] — 기술적 지표 상세
- [[overview]] — 프로젝트 개요

---

[[index|← 인덱스로 돌아가기]]
