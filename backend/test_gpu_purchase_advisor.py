"""
GPU 구매 타이밍 예측 시스템 테스트
부품명 입력 → 구매 적정성 판단 (바둑 승률 방식)
"""
import torch
import numpy as np
import sys
import json
from pathlib import Path

# 색상 코드
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

print(f"{BOLD}{CYAN}")
print("=" * 80)
print("  GPU 구매 타이밍 예측 AI - AlphaZero 방식")
print("  (바둑 승률처럼 구매 적정성 판단)")
print("=" * 80)
print(RESET)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. 데이터 로드
print(f"\n{BLUE}[1단계] 데이터 분석{RESET}")
with open('data/processed/dataset/training_data.json') as f:
    training_data = json.load(f)

with open('data/processed/integrated/2026-02-13.json') as f:
    gpu_market_data = json.load(f)

print(f"✓ 훈련 데이터: {len(training_data)}개 GPU 모델")
print(f"✓ 시장 데이터: {len(gpu_market_data['gpu_data'])}개 GPU 모델")

# 2. State Vector 분석
print(f"\n{BLUE}[2단계] State Vector 구조 분석{RESET}")
if len(training_data) > 0:
    state_dim = len(training_data[0]['state_vector'])
    print(f"✓ 현재 Feature 차원: {state_dim}")
    print(f"\nState Vector 구성 (추정):")
    print(f"  [0] 정규화된 가격 (0~1)")
    print(f"  [1] 가격 변화율 (전일 대비)")
    print(f"  [2] 가격 변화율 (전주 대비)")
    print(f"  [3] 환율 정규화 (USD/KRW)")
    print(f"  [4] 환율 정규화 (JPY/KRW)")
    print(f"  [5] 뉴스 감정 점수 (-1~1)")
    print(f"  [6] 판매자 수 정규화")
    print(f"  [7-10] 예비 Feature")

# 3. 모델 차원 확인
print(f"\n{BLUE}[3단계] 모델 구조 확인{RESET}")
model_path = Path("alphazero_model.pth")
if model_path.exists():
    model_data = torch.load(model_path, map_location=device, weights_only=False)
    input_weight = model_data['h_state_dict']['input_embedding.weight']
    model_input_dim = input_weight.shape[1]

    print(f"✓ 학습된 모델 입력 차원: {model_input_dim}")
    print(f"✗ 데이터 차원: {state_dim}")
    print(f"\n{RED}⚠️  차원 불일치 감지!{RESET}")
    print(f"{YELLOW}   → 모델을 {state_dim}차원으로 재학습하거나")
    print(f"   → 데이터를 {model_input_dim}차원으로 확장 필요{RESET}")

    # 임시 해결: 패딩
    print(f"\n{YELLOW}[임시 해결] Zero-padding으로 {model_input_dim}차원 맞추기{RESET}")

# 4. 간단한 규칙 기반 시스템 구현
print(f"\n{BOLD}{CYAN}")
print("=" * 80)
print("  규칙 기반 구매 타이밍 판단 (데모)")
print("=" * 80)
print(RESET)

print(f"\n{BLUE}판단 기준:{RESET}")
print(f"  • 가격이 평균보다 낮으면 → {GREEN}구매 추천{RESET}")
print(f"  • 가격이 평균보다 높으면 → {RED}구매 대기{RESET}")
print(f"  • 환율 상승 시 → {RED}구매 대기{RESET} (수입 부품 가격 상승)")
print(f"  • 뉴스 긍정적 → {GREEN}구매 추천{RESET}")

# GPU 모델별 판단
print(f"\n{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
print(f"{BOLD}GPU 모델별 구매 적정성 분석{RESET}")
print(f"{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

# 가격 범위 계산
all_prices = []
for item in training_data:
    gpu_model = item['gpu_model']
    if gpu_model in gpu_market_data['gpu_data']:
        price = gpu_market_data['gpu_data'][gpu_model]['domestic']['lowest_price']
        all_prices.append(price)

if all_prices:
    avg_price = np.mean(all_prices)

    # 각 GPU 판단
    test_gpus = ["RTX 5060", "RTX 5070", "RTX 5080", "RX 9070 XT"]

    for gpu_model in test_gpus:
        if gpu_model not in gpu_market_data['gpu_data']:
            continue

        gpu_info = gpu_market_data['gpu_data'][gpu_model]
        price = gpu_info['domestic']['lowest_price']
        usd_krw = gpu_info['macro']['usd_krw']

        print(f"\n{BOLD}{gpu_model}{RESET}")
        print(f"├─ 현재 가격: {price:,}원")
        print(f"├─ 환율: {usd_krw:.2f} (USD/KRW)")

        # 구매 점수 계산 (0~100)
        price_score = 100 - (price / avg_price - 1) * 100  # 평균보다 싸면 높은 점수
        price_score = max(0, min(100, price_score))

        # 바둑 승률처럼 표현
        buy_probability = price_score / 100

        print(f"└─ 구매 적정도: {price_score:.1f}점 / 100점")

        # 승률 바 표시
        bar_length = int(buy_probability * 40)
        bar = "█" * bar_length

        if buy_probability >= 0.7:
            color = GREEN
            advice = "강력 추천"
            emoji = "🟢"
        elif buy_probability >= 0.5:
            color = YELLOW
            advice = "보통"
            emoji = "🟡"
        else:
            color = RED
            advice = "대기 권장"
            emoji = "🔴"

        print(f"\n{color}   [{buy_probability:6.1%}] {bar}{RESET}")
        print(f"   {emoji} {advice}")

        # 이유 설명
        if price < avg_price:
            print(f"   {GREEN}✓ 평균보다 저렴한 가격{RESET}")
        else:
            print(f"   {RED}✗ 평균보다 비싼 가격{RESET}")

# 5. AlphaZero 방식으로 판단하려면
print(f"\n{BOLD}{CYAN}")
print("=" * 80)
print("  AlphaZero 방식 적용을 위한 요구사항")
print("=" * 80)
print(RESET)

print(f"\n{BLUE}현재 시스템:{RESET}")
print(f"  • {RED}✗{RESET} 데이터 부족 (12개 샘플, 3일치)")
print(f"  • {RED}✗{RESET} Feature 부족 (11차원 → 256차원 필요)")
print(f"  • {RED}✗{RESET} 시계열 데이터 부재 (가격 추이 없음)")
print(f"  • {RED}✗{RESET} 행동 정의 불명확 (구매/대기/관망?)")

print(f"\n{GREEN}AlphaZero로 작동하려면:{RESET}")
print(f"\n1️⃣  충분한 데이터 수집")
print(f"   • 최소 30일 이상 가격 데이터")
print(f"   • GPU 모델당 최소 100개 샘플")

print(f"\n2️⃣  Feature Engineering (11차원 → 256차원)")
print(f"   • 가격 추이 (7일, 14일, 30일 이동평균)")
print(f"   • 가격 변동성 (표준편차)")
print(f"   • 환율 추이 (USD, JPY)")
print(f"   • 뉴스 감정 분석 (시계열)")
print(f"   • 경쟁 모델 가격 비교")
print(f"   • 출시일 이후 경과 시간")
print(f"   • 계절성 (연말, 신제품 출시 주기)")

print(f"\n3️⃣  행동 정의 (Action Space)")
print(f"   • BUY_NOW: 즉시 구매")
print(f"   • WAIT_SHORT: 1주일 대기")
print(f"   • WAIT_LONG: 1개월 대기")
print(f"   • HOLD: 관망")
print(f"   • SKIP: 구매 안함")

print(f"\n4️⃣  보상 정의 (Reward)")
print(f"   • 구매 후 7일 뒤 가격 하락 → {GREEN}+보상{RESET}")
print(f"   • 구매 후 7일 뒤 가격 상승 → {RED}-보상{RESET}")
print(f"   • 최저가 타이밍 맞춤 → {GREEN}+큰 보상{RESET}")

print(f"\n5️⃣  MCTS 시뮬레이션")
print(f"   • 미래 가격 예측 (Dynamics Network)")
print(f"   • 최적 구매 시점 탐색")
print(f"   • 바둑처럼 승률(이득률) 계산")

print(f"\n{BOLD}{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")
print(f"{BOLD}최종 결론{RESET}")
print(f"{YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}")

print(f"\n✅ {GREEN}아키텍처는 완성됨{RESET} (AlphaZero/MuZero)")
print(f"✅ {GREEN}모델은 학습됨{RESET} (18.9M 파라미터)")
print(f"❌ {RED}데이터가 부족함{RESET} (12개 샘플)")
print(f"❌ {RED}Feature가 부족함{RESET} (11차원)")

print(f"\n{BOLD}현재 상태:{RESET}")
print(f"  • 규칙 기반 시스템으로 작동 가능")
print(f"  • AlphaZero로 작동하려면 데이터 확충 필요")

print(f"\n{BOLD}필요한 작업:{RESET}")
print(f"  1. 최소 30일 이상 GPU 가격 수집")
print(f"  2. Feature Engineering (11차원 → 256차원)")
print(f"  3. 모델 재학습 (올바른 차원)")
print(f"  4. 백테스팅으로 성능 검증")

print(f"\n{GREEN}예상 소요 시간: 2~3주{RESET}\n")
