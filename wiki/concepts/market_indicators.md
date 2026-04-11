# 시장 지표 및 기술적 분석

> 마지막 업데이트: 2026-04-11

## 개요

Feature Engineering의 106차원 기술적 지표 부분을 구성하는 핵심 지표들입니다.

## 주요 기술적 지표

### RSI (Relative Strength Index)
- 과매수/과매도 상태를 판단
- 70 이상: 과매수 (가격 하락 가능)
- 30 이하: 과매도 (가격 상승 가능)

### MACD (Moving Average Convergence Divergence)
- 단기/장기 이동평균의 차이로 추세 파악
- 시그널 라인 교차로 매수/매도 타이밍 판단

### Bollinger Bands
- 이동평균 ± 2표준편차
- 밴드 폭으로 변동성 측정
- 밴드 돌파 시 추세 전환 신호

## 시장 관련 지표

- 재고 상태 (in_stock / low_stock / out_of_stock)
- 판매자 수 변화
- 환율 영향 (USD/KRW, JPY/KRW, EUR/KRW)

## 관련 페이지

- [[concepts/feature_engineering]] — 전체 특징 공학 파이프라인
- [[sources/danawa_data]] — 가격/재고 데이터
- [[sources/exchange_data]] — 환율 데이터

---

[[index|← 인덱스로 돌아가기]]
