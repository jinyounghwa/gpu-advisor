"""
Wiki 자동 업데이트 스크립트
일일 크롤링 완료 후 wiki/ 디렉토리의 페이지들을 자동 갱신합니다.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WIKI_DIR_NAME = "wiki"


def _wiki_dir(project_root: Path) -> Path:
    return project_root / WIKI_DIR_NAME


def _ensure_dirs(project_root: Path) -> None:
    """wiki/ 하위 디렉토리가 존재하는지 확인하고 생성."""
    wiki = _wiki_dir(project_root)
    for sub in ("gpus", "concepts", "analysis", "sources"):
        (wiki / sub).mkdir(parents=True, exist_ok=True)


def _format_price(price: int) -> str:
    """가격을 읽기 편한 형식으로 포맷."""
    return f"{price:,}원"


def _stock_label(status: str) -> str:
    mapping = {
        "in_stock": "충분",
        "low_stock": "저재고",
        "out_of_stock": "품절",
    }
    return mapping.get(status, status)


def _read_json_safe(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _get_previous_day_data(raw_dir: Path, source: str, today: str) -> Optional[Dict]:
    """어제 데이터를 찾아서 반환 (가격 변동 비교용)."""
    from datetime import timedelta

    today_dt = datetime.strptime(today, "%Y-%m-%d")
    yesterday = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    return _read_json_safe(raw_dir / source / f"{yesterday}.json")


def _compute_price_change(current_price: int, previous_price: Optional[int]) -> Optional[str]:
    if previous_price is None:
        return None
    diff = current_price - previous_price
    pct = (diff / previous_price) * 100 if previous_price > 0 else 0
    if diff > 0:
        return f"▲ +{_format_price(diff)} ({pct:+.1f}%)"
    elif diff < 0:
        return f"▼ {_format_price(diff)} ({pct:+.1f}%)"
    return "→ 변동 없음"


def update_gpu_pages(
    project_root: Path,
    today: str,
    danawa_data: Dict[str, Any],
    prev_danawa: Optional[Dict[str, Any]],
) -> List[str]:
    """GPU 모델별 위키 페이지를 갱신하고 업데이트된 파일 목록 반환."""
    wiki = _wiki_dir(project_root)
    gpus_dir = wiki / "gpus"
    gpus_dir.mkdir(parents=True, exist_ok=True)

    prev_products = {}
    if prev_danawa:
        for p in prev_danawa.get("products", []):
            prev_products[p.get("chipset", "")] = p

    updated_files = []
    for product in danawa_data.get("products", []):
        chipset = product.get("chipset", "")
        if not chipset:
            continue

        safe_name = chipset.replace(" ", "_")
        page_path = gpus_dir / f"{safe_name}.md"

        price = product.get("lowest_price", 0)
        prev_price = prev_products.get(chipset, {}).get("lowest_price")
        change_str = _compute_price_change(price, prev_price)

        # 이전 페이지 읽기 (히스토리 유지)
        existing_content = ""
        if page_path.exists():
            existing_content = page_path.read_text(encoding="utf-8")

        # 히스토리 섹션 추출: "### 이전 기록" 이하의 내용만 보존
        history_lines = []
        in_history = False
        for line in existing_content.split("\n"):
            if line.strip() == "### 이전 기록":
                in_history = True
                continue
            if in_history:
                if line.startswith("## ") or line.startswith("# "):
                    in_history = False
                elif line.strip() == "---" or line.startswith("[["):
                    in_history = False
                else:
                    history_lines.append(line)

        lines = [
            f"# {chipset}",
            "",
            f"> 마지막 업데이트: {today}",
            "",
            "## 기본 정보",
            "",
            f"| 항목 | 값 |",
            f"|------|-----|",
            f"| 칩셋 | {chipset} |",
            f"| 제조사 | {product.get('manufacturer', 'N/A')} |",
            f"| 최저가 | {_format_price(price)} |",
            f"| 전일 대비 | {change_str or 'N/A (첫 수집)'} |",
            f"| 판매자 수 | {product.get('seller_count', 0)}명 |",
            f"| 재고 상태 | {_stock_label(product.get('stock_status', 'unknown'))} |",
            "",
            "## 상세",
            "",
            f"- **제품명**: {product.get('product_name', 'N/A')}",
            f"- **다나와 링크**: [제품 페이지]({product.get('product_url', '#')})",
            "",
        ]

        # 가격 히스토리
        lines.append("## 가격 히스토리")
        lines.append("")
        lines.append(f"| 날짜 | {today} |")
        lines.append(f"|------|--------|")
        lines.append(f"| 최저가 | {_format_price(price)} |")
        if change_str:
            lines.append(f"| 변동 | {change_str} |")

        # 기존 히스토리 추가
        if history_lines:
            lines.append("")
            lines.append("### 이전 기록")
            for hl in history_lines:
                if hl.strip():
                    lines.append(hl)

        lines.append("")
        lines.append("---")
        lines.append(f"[[index|← 인덱스로 돌아가기]]")

        page_path.write_text("\n".join(lines), encoding="utf-8")
        updated_files.append(f"gpus/{safe_name}.md")

    return updated_files


def update_analysis_pages(
    project_root: Path,
    today: str,
    danawa_data: Dict[str, Any],
    exchange_data: Optional[Dict[str, Any]],
    news_data: Optional[Dict[str, Any]],
) -> List[str]:
    """분석 페이지 (price_trends, market_sentiment) 갱신."""
    wiki = _wiki_dir(project_root)
    analysis_dir = wiki / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    updated = []

    # --- 가격 트렌드 ---
    products = danawa_data.get("products", [])
    if products:
        lines = [
            "# GPU 가격 트렌드 분석",
            "",
            f"> 마지막 업데이트: {today}",
            "",
            "## 전체 모델 가격 현황",
            "",
            "| 모델 | 최저가 | 재고 | 판매자 |",
            "|------|--------|------|--------|",
        ]
        for p in sorted(products, key=lambda x: x.get("lowest_price", 0)):
            lines.append(
                f"| {p.get('chipset', '?')} "
                f"| {_format_price(p.get('lowest_price', 0))} "
                f"| {_stock_label(p.get('stock_status', '?'))} "
                f"| {p.get('seller_count', 0)} |"
            )

        lines.extend(
            [
                "",
                "## 주요 관찰",
                "",
            ]
        )

        # 품절 모델
        oos = [p for p in products if p.get("stock_status") == "out_of_stock"]
        if oos:
            oos_names = ", ".join(p.get("chipset", "?") for p in oos)
            lines.append(f"- **품절 모델** ({len(oos)}개): {oos_names}")

        # 저재고
        low = [p for p in products if p.get("stock_status") == "low_stock"]
        if low:
            low_names = ", ".join(p.get("chipset", "?") for p in low)
            lines.append(f"- **저재고 모델** ({len(low)}개): {low_names}")

        budget = [p for p in products if 0 < p.get("lowest_price", 0) <= 1_000_000]
        if budget:
            budget_names = ", ".join(
                f"{p.get('chipset', '?')}({_format_price(p.get('lowest_price', 0))})"
                for p in budget
            )
            lines.append(f"- **100만원 이하** ({len(budget)}개): {budget_names}")

        lines.extend(["", "---", "[[index|← 인덱스로 돌아가기]]", ""])

        path = analysis_dir / "price_trends.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        updated.append("analysis/price_trends.md")

    # --- 시장 감성 ---
    if news_data:
        stats = news_data.get("statistics", {})
        lines = [
            "# 시장 감성 분석",
            "",
            f"> 마지막 업데이트: {today}",
            "",
            "## 감성 분포",
            "",
            f"| 감성 | 기사 수 |",
            f"|------|---------|",
            f"| 긍정 | {stats.get('positive_count', 0)} |",
            f"| 부정 | {stats.get('negative_count', 0)} |",
            f"| 중립 | {stats.get('neutral_count', 0)} |",
            f"| **총합** | {stats.get('total', 0)} |",
            "",
            f"**평균 감성 점수**: {stats.get('sentiment_avg', 0):.2f} "
            f"({'긍정' if stats.get('sentiment_avg', 0) > 0.1 else '부정' if stats.get('sentiment_avg', 0) < -0.1 else '중립'})",
            "",
            "## 주요 뉴스 헤드라인",
            "",
        ]
        for article in news_data.get("articles", [])[:10]:
            sentiment_icon = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(
                article.get("sentiment", "neutral"), "⚪"
            )
            lines.append(
                f"- {sentiment_icon} [{article.get('source', '?')}] {article.get('title', 'N/A')}"
            )

        lines.extend(["", "---", "[[index|← 인덱스로 돌아가기]]", ""])

        path = analysis_dir / "market_sentiment.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        updated.append("analysis/market_sentiment.md")

    return updated


def update_index(project_root: Path, today: str, danawa_data: Dict[str, Any]) -> str:
    """index.md 갱신."""
    wiki = _wiki_dir(project_root)
    products = danawa_data.get("products", [])

    lines = [
        "# GPU Advisor Wiki — 인덱스",
        "",
        f"> 마지막 업데이트: {today}",
        "",
        "## 개요",
        "",
        "| 페이지 | 요약 | 업데이트 |",
        "|--------|------|----------|",
        "| [[overview]] | 프로젝트 전체 개요 | " + today + " |",
        "",
        "---",
        "",
        "## GPU 모델",
        "",
        "| 페이지 | 최저가 | 재고 | 업데이트 |",
        "|--------|--------|------|----------|",
    ]
    for p in sorted(products, key=lambda x: x.get("lowest_price", 0)):
        chipset = p.get("chipset", "?")
        safe_name = chipset.replace(" ", "_")
        lines.append(
            f"| [[gpus/{safe_name}|{chipset}]] "
            f"| {_format_price(p.get('lowest_price', 0))} "
            f"| {_stock_label(p.get('stock_status', '?'))} "
            f"| {today} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 개념",
            "",
            "| 페이지 | 요약 | 업데이트 |",
            "|--------|------|----------|",
            "| [[concepts/mcts]] | 몬테카를로 트리 탐색 — 의사결정 탐색 알고리즘 | " + today + " |",
            "| [[concepts/alphazero]] | AlphaZero/MuZero 아키텍처 (h/g/f + MCTS) | " + today + " |",
            "| [[concepts/feature_engineering]] | 256차원 특징 공학 파이프라인 | " + today + " |",
            "| [[concepts/market_indicators]] | 시장 지표 및 기술적 분석 지표 | " + today + " |",
            "",
            "---",
            "",
            "## 분석",
            "",
            "| 페이지 | 요약 | 업데이트 |",
            "|--------|------|----------|",
            "| [[analysis/price_trends]] | 전체 GPU 가격 트렌드 분석 | " + today + " |",
            "| [[analysis/market_sentiment]] | 뉴스 감성 분석 종합 | " + today + " |",
            "",
            "---",
            "",
            "## 소스",
            "",
            "| 페이지 | 요약 | 업데이트 |",
            "|--------|------|----------|",
            "| [[sources/danawa_data]] | 다나와 가격 크롤러 — 24개 모델 추적 | " + today + " |",
            "| [[sources/exchange_data]] | 환율 수집 (USD/KRW, JPY/KRW, EUR/KRW) | " + today + " |",
            "| [[sources/news_data]] | Google News RSS 뉴스 크롤러 + 감성 분석 | " + today + " |",
            "",
        ]
    )

    path = wiki / "index.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return "index.md"


def ensure_concept_pages(project_root: Path, today: str) -> List[str]:
    """기본 개념 페이지들이 존재하는지 확인하고 생성."""
    wiki = _wiki_dir(project_root)
    concepts_dir = wiki / "concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)
    updated = []

    concepts = {
        "mcts.md": (
            "# 몬테카를로 트리 탐색 (MCTS)",
            "의사결정 트리를 구축하며 확률적 시뮬레이션을 통해 최적 행동을 탐색하는 알고리즘."
            " AlphaGo의 핵심 기술.",
        ),
        "alphazero.md": (
            "# AlphaZero / MuZero 아키텍처",
            "강화학습의 혁신적 아키텍처. h(Representation) + g(Dynamics) + f(Prediction) + MCTS"
            "로 구성하여 모델 기반 의사결정을 수행.",
        ),
        "feature_engineering.md": (
            "# 256차원 특징 공학",
            "GPU 시장 데이터(가격, 트렌드, 뉴스 감성, 환율)을 256차원 벡터로 변환하는 과정."
            " 신경망의 입력 표현.",
        ),
        "market_indicators.md": (
            "# 시장 지표 및 기술적 분석",
            "GPU 시장 상황을 판단하기 위한 지표: 가격 추세, 재고 수준, 뉴스 감성, 환율 변동 등.",
        ),
    }

    for filename, (title, desc) in concepts.items():
        page_path = concepts_dir / filename
        if not page_path.exists():
            lines = [
                title,
                "",
                f"> 마지막 업데이트: {today}",
                "",
                "## 개요",
                "",
                desc,
                "",
                "---",
                f"[[../index|← 인덱스로 돌아가기]]",
                "",
            ]
            page_path.write_text("\n".join(lines), encoding="utf-8")
            updated.append(f"concepts/{filename}")

    return updated


def ensure_source_pages(project_root: Path, today: str) -> List[str]:
    """데이터 소스 설명 페이지들이 존재하는지 확인하고 생성."""
    wiki = _wiki_dir(project_root)
    sources_dir = wiki / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    updated = []

    sources = {
        "danawa_data.md": (
            "# 다나와 가격 데이터",
            "다나와(Danawa)에서 일일 수집하는 GPU 가격 정보. 한국 시장 최저가 추적.",
        ),
        "exchange_data.md": (
            "# 환율 데이터",
            "해외 GPU 수입 영향을 고려한 환율 정보. USD/KRW, JPY/KRW, EUR/KRW 추적.",
        ),
        "news_data.md": (
            "# GPU 뉴스 및 감성 분석",
            "Google News RSS에서 수집한 GPU 관련 뉴스. 자동 감성 분석으로 시장 분위기 파악.",
        ),
    }

    for filename, (title, desc) in sources.items():
        page_path = sources_dir / filename
        if not page_path.exists():
            lines = [
                title,
                "",
                f"> 마지막 업데이트: {today}",
                "",
                "## 개요",
                "",
                desc,
                "",
                "---",
                f"[[../index|← 인덱스로 돌아가기]]",
                "",
            ]
            page_path.write_text("\n".join(lines), encoding="utf-8")
            updated.append(f"sources/{filename}")

    return updated


def update_overview(project_root: Path, today: str) -> str:
    """overview.md 갱신 (항상 최신 상태 유지)."""
    wiki = _wiki_dir(project_root)
    overview_path = wiki / "overview.md"

    lines = [
        "# GPU Advisor — 프로젝트 개요",
        "",
        f"> 마지막 업데이트: {today}",
        "",
        "## 프로젝트 목적",
        "",
        "**\"이 GPU를 지금 살까, 기다릴까?\"** — AlphaZero/MuZero 스타일 강화학습으로 GPU 최적 구매 타이밍을 결정하는 AI 에이전트.",
        "",
        "## 핵심 원리",
        "",
        "Go에서 AlphaGo가 \"이 수를 둘까?\"를 평가하듯, 이 시스템은 \"이 GPU를 지금 살까?\"를 평가합니다:",
        "- **상태 표현**: 시장 조건(가격, 트렌드, 뉴스 감성) → 256차원 잠재 상태",
        "- **MCTS 시뮬레이션**: 50개 미래 시나리오 시뮬레이션 (가격 하락, 신제품 출시, 시장 충격)",
        "- **가치 예측**: 각 시나리오의 구매 타이밍 최적성 평가",
        "- **최종 추천**: 시뮬레이션 결과 기반 최적 행동(Buy Now / Wait) 선택",
        "",
        "---",
        f"[[index|← 인덱스로 돌아가기]]",
        "",
    ]

    overview_path.write_text("\n".join(lines), encoding="utf-8")
    return "overview.md"


def append_log(project_root: Path, today: str, updated_files: List[str]) -> str:
    """log.md에 오늘 작업 내역을 추가."""
    wiki = _wiki_dir(project_root)
    log_path = wiki / "log.md"

    existing = ""
    if log_path.exists():
        existing = log_path.read_text(encoding="utf-8")

    entry_lines = [
        f"## [{today}] ingest | 일일 데이터 위키 반영",
        "",
        f"- 업데이트된 페이지: {len(updated_files)}개",
    ]
    for f in updated_files:
        entry_lines.append(f"  - `{f}`")
    entry_lines.append("")

    # 파일 끝에 추가
    new_content = existing.rstrip() + "\n\n" + "\n".join(entry_lines) + "\n"
    log_path.write_text(new_content, encoding="utf-8")
    return "log.md"


def run_wiki_update(
    project_root: Path,
    step_results: Dict[str, bool],
) -> Dict[str, Any]:
    """
    일일 크롤링 결과를 wiki/에 반영.

    Returns:
        dict with keys: success, updated_files, today, error
    """
    today = datetime.now().strftime("%Y-%m-%d")
    raw_dir = project_root / "data" / "raw"
    wiki = _wiki_dir(project_root)

    _ensure_dirs(project_root)

    result: Dict[str, Any] = {
        "success": False,
        "updated_files": [],
        "today": today,
        "error": None,
    }

    try:
        # 데이터 로드
        danawa_data = _read_json_safe(raw_dir / "danawa" / f"{today}.json")
        exchange_data = _read_json_safe(raw_dir / "exchange" / f"{today}.json")
        news_data = _read_json_safe(raw_dir / "news" / f"{today}.json")

        if not danawa_data:
            result["error"] = f"danawa data not found for {today}"
            logger.warning(f"위키 업데이트 건너뜀: {result['error']}")
            return result

        prev_danawa = _get_previous_day_data(raw_dir, "danawa", today)

        updated: List[str] = []

        # 1) GPU 페이지 갱신
        gpu_files = update_gpu_pages(project_root, today, danawa_data, prev_danawa)
        updated.extend(gpu_files)
        logger.info(f"  위키 GPU 페이지 갱신: {len(gpu_files)}개")

        # 2) 분석 페이지 갱신
        analysis_files = update_analysis_pages(
            project_root, today, danawa_data, exchange_data, news_data
        )
        updated.extend(analysis_files)
        logger.info(f"  위키 분석 페이지 갱신: {len(analysis_files)}개")

        # 3) 개념 페이지 보장
        concept_files = ensure_concept_pages(project_root, today)
        updated.extend(concept_files)
        if concept_files:
            logger.info(f"  위키 개념 페이지 생성: {len(concept_files)}개")

        # 4) 소스 페이지 보장
        source_files = ensure_source_pages(project_root, today)
        updated.extend(source_files)
        if source_files:
            logger.info(f"  위키 소스 페이지 생성: {len(source_files)}개")

        # 5) overview.md 갱신
        overview = update_overview(project_root, today)
        updated.append(overview)
        logger.info(f"  위키 개요 갱신: {overview}")

        # 6) index.md 갱신
        idx = update_index(project_root, today, danawa_data)
        updated.append(idx)
        logger.info(f"  위키 인덱스 갱신: {idx}")

        # 7) log.md 추가
        log = append_log(project_root, today, updated)
        updated.append(log)
        logger.info(f"  위키 로그 갱신: {log}")

        result["success"] = True
        result["updated_files"] = updated

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"위키 업데이트 실패: {e}", exc_info=True)

    return result
