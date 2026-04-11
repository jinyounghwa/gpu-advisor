# MCTS (Monte Carlo Tree Search)

> 마지막 업데이트: 2026-04-11

## 개요

MCTS는 의사결정 문제에서 최적의 행동을 찾기 위한 휴리스틱 탐색 알고리즘입니다. GPU Advisor에서는 **"지금 구매할까, 기다릴까?"**를 결정하기 위해 사용됩니다.

## 작동 원리

각 시뮬레이션은 4단계로 구성됩니다:

1. **Selection (선택)**: 트리의 루트에서 시작, UCB 점수가 가장 높은 자식 노드를 따라 내려감
2. **Expansion (확장)**: 리프 노드에서 아직 시도하지 않은 행동을 추가
3. **Simulation (롤아웃)**: 새 노드에서 정책 네트워크를 사용해 미래 시나리오 전개
4. **Backpropagation (역전파)**: 시뮬레이션 결과값을 부모 노드들에 전파

## GPU Advisor 설정

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 시뮬레이션 수 | 50 | 의사결정당 시뮬레이션 횟수 |
| PUCT 상수 | 탐색 | 상-하위 균형 |
| Dirichlet noise | ε=0.25, α=0.03 | 탐색 다양성 |
| Rollout 깊이 | 5 | 미래 예측 스텝 |

## 관련 페이지

- [[concepts/alphazero]] — AlphaZero 전체 아키텍처
- [[concepts/feature_engineering]] — MCTS 입력값 생성
- [[overview]] — 프로젝트 개요

---

[[index|← 인덱스로 돌아가기]]
