"""
Centralized reward calculation for the GPU advisor agent.
Used by both the evaluator and fine-tuner to ensure consistency.
"""

from typing import Union


def calculate_reward(action: Union[int, str], pct_change: float) -> float:
    """
    행동 조건부 보상 함수.

    action은 문자열("BUY_NOW", "WAIT_SHORT", ...) 또는 정수 인덱스(0, 1, ...)가 될 수 있습니다.
    
    BUY_NOW (0)  → pct_change          : 가격이 오르면 매수가 맞았음 (양수)
    WAIT_SHORT (1), WAIT_LONG (2) → -pct_change : 가격이 내리면 대기가 맞았음 (양수)
    HOLD (3)     → -abs(pct_change) * 0.1     : 불확실성 비용 최소
    SKIP (4)     → -abs(pct_change) * 0.15    : 기회비용 (HOLD보다 소폭 높음)
    """
    # Convert string action to standard format
    if isinstance(action, str):
        if action == "BUY_NOW":
            return pct_change
        if action in {"WAIT_SHORT", "WAIT_LONG"}:
            return -pct_change
        if action == "HOLD":
            return -abs(pct_change) * 0.1
        if action == "SKIP":
            return -abs(pct_change) * 0.15
        return 0.0

    # Handle integer action indices
    if action == 0:  # BUY_NOW
        return pct_change
    if action in {1, 2}:  # WAIT_SHORT, WAIT_LONG
        return -pct_change
    if action == 3:  # HOLD
        return -abs(pct_change) * 0.1
    if action == 4:  # SKIP
        return -abs(pct_change) * 0.15
    return 0.0
