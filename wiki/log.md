# GPU Advisor Wiki — 작업 로그

> 이 파일은 위키의 변경 이력을 시간순으로 기록합니다.
> 검색 팁: `grep "^## \[" log.md | tail -5` 로 최근 5개 항목 확인

---

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

