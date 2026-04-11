# 다나와 GPU 가격 데이터

> 마지막 업데이트: 2026-04-11

## 개요

다나와(danawa.com)에서 매일 수집하는 GPU 가격 정보입니다.

## 수집 대상

24개 GPU 모델:
- NVIDIA: RTX 5090, 5080, 5070 Ti, 5070, 5060 Ti, 5060, 5050, RTX 4090, 4080, 4070 Ti, 4070, 4060 Ti, 4060
- AMD: RX 9070 XT, 9060 XT, 7900 XTX, 7900 XT, 7800 XT, 7700 XT, 7600, 6600
- Intel: Arc B580, B570, A770

## 수집 항목

| 항목 | 설명 |
|------|------|
| product_name | 제품명 |
| manufacturer | 제조사 |
| chipset | 칩셋 모델 |
| lowest_price | 최저가 (원) |
| seller_count | 판매자 수 |
| stock_status | 재고 상태 (in_stock/low_stock/out_of_stock) |
| product_url | 다나와 제품 페이지 URL |

## 저장 위치

`data/raw/danawa/YYYY-MM-DD.json`

## 크롤러

`crawlers/danawa_crawler.py` — BeautifulSoup 기반, 1초 간격 요청

---

[[index|← 인덱스로 돌아가기]]
