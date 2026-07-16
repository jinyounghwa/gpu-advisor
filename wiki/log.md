# GPU Advisor Wiki — 작업 로그

> 이 파일은 위키의 변경 이력을 시간순으로 기록합니다.
> 검색 팁: `grep "^## \[" log.md | tail -5` 로 최근 5개 항목 확인

---

## [2026-05-26] update | AlphaZero 모델 수동 강제 재학습 완료

- 95일간 축적된 실제 마켓 데이터를 바탕으로 AlphaZero 모델 강제 재학습(2000 steps)을 수행함
- 평가 데이터 윈도우: 95일 확보 (2026-02-21 ~ 2026-05-26)
- 7개 품질 게이트 검증: PASS
- 주요 지표 성과:
  - 방향 정확도 (Directional Accuracy): 92.99%
  - 평균 의사결정 보상 (Avg Reward): +0.00588
  - 관망 비율 (Abstain Ratio): 83.91%
  - 액션 엔트로피 (Action Entropy): 0.6040
- 영향받은 페이지: `index.md`, `log.md`

## [2026-04-11] initialize | 위키 초기 생성

- wiki/ 디렉토리 구조 생성 (gpus/, concepts/, analysis/, sources/)
- index.md, log.md, overview.md 초기 페이지 작성
- GPU 모델별 페이지 21개 생성 (data/raw/danawa/2026-04-11.json 기반)
- 개념 페이지 4개 생성 (MCTS, AlphaZero, Feature Engineering, Market Indicators)
- 분석 페이지 2개 생성 (Price Trends, Market Sentiment)
- 소스 페이지 3개 생성 (Danawa, Exchange, News)
- AGENTS.md Schema 파일 생성
- crawlers/wiki_updater.py 자동 업데이트 스크립트 작성
- run_daily.py에 wiki_updater 연동
- 일일 보고서(status_report.py)에 위키 반영 내역 섹션 추가

## [2026-04-11] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 24개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `index.md`

## [2026-04-11] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-12] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 22개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-13] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 24개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-14] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-15] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-16] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-17] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-18] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 23개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-19] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 23개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-20] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 23개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-21] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 23개
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-21] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 22개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-22] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 22개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-23] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-24] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-25] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-26] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-27] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 22개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-28] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 22개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-29] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-04-30] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-01] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-02] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-03] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-04] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-05] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 24개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-06] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 24개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-07] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 24개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-08] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-09] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-10] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-11] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-12] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-13] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`


## [2026-05-15] lint | 프로젝트 불필요 파일 정리

- **캐시/임시 파일 삭제**: `__pycache__/` (전체), `.pytest_cache/`, `.ruff_cache/`, `.DS_Store`, `frontend_pid.txt`, `gpu_advisor.egg-info/`
- **구형 모델 삭제**: `alphazero_model.pth` (72MB 초기 체크포인트, `alphazero_model_agent_latest.pth`로 대체됨)
- **빈 Obsidian vault 삭제**: `wiki/gpuadvicewiki/` (Welcome.md만 있는 빈 템플릿)
- **루트 중복/구형 문서 삭제**:
  - `wiki.md` → `wiki/` 디렉토리로 통합됨
  - `feasibility_report.md` → 초기 타당성 검토 (내용 구버전)
  - `GPU_PURCHASE_ADVISOR_REPORT.md` → 2026-02-14 초기 평가 보고서
  - `종합_프로젝트_보고서.md` → 초기 종합 보고서
- `.gitignore`에 삭제 파일 패턴 추가 (재생성 방지)
- 영향받은 페이지: `log.md`, `index.md`

## [2026-05-17] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-18] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-19] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-20] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-21] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-22] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-23] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-24] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-25] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-26] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-27] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-28] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-29] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-30] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-05-31] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-01] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-02] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-03] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 25개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-04] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-05] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-06] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-07] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-08] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-09] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-10] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-11] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-12] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-13] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-14] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-15] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-16] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-17] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-18] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-19] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-20] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-21] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-22] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-23] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-24] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-25] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-26] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-27] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-28] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-29] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-06-30] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-01] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-02] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-03] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-04] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-05] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-06] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-07] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-08] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-09] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-10] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-11] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-12] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-13] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-14] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-15] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

## [2026-07-16] ingest | 일일 데이터 위키 반영

- 업데이트된 페이지: 26개
  - `gpus/RTX_5090.md`
  - `gpus/RTX_5080.md`
  - `gpus/RTX_5070_Ti.md`
  - `gpus/RTX_5070.md`
  - `gpus/RTX_5060_Ti.md`
  - `gpus/RTX_5060.md`
  - `gpus/RTX_5050.md`
  - `gpus/RTX_4090.md`
  - `gpus/RTX_4080.md`
  - `gpus/RTX_4070_Ti.md`
  - `gpus/RTX_4070.md`
  - `gpus/RTX_4060_Ti.md`
  - `gpus/RTX_4060.md`
  - `gpus/RX_9070_XT.md`
  - `gpus/RX_9060_XT.md`
  - `gpus/RX_7900_XTX.md`
  - `gpus/RX_7800_XT.md`
  - `gpus/RX_7700_XT.md`
  - `gpus/RX_7600.md`
  - `gpus/RX_6600.md`
  - `gpus/Arc_B580.md`
  - `gpus/Arc_A770.md`
  - `analysis/price_trends.md`
  - `analysis/market_sentiment.md`
  - `overview.md`
  - `index.md`

