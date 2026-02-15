#!/bin/bash
###############################################################################
# GPU 구매 타이밍 AI - Cron 자동화 설정 스크립트
# 매일 자정에 크롤링 실행
###############################################################################

# 색상 코드
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  GPU 구매 타이밍 AI - Cron 자동화 설정"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 현재 디렉토리 절대 경로
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo -e "\n${GREEN}✓${NC} 프로젝트 경로: $PROJECT_DIR"

# Python 경로 찾기
PYTHON_PATH=$(which python3)
echo -e "${GREEN}✓${NC} Python 경로: $PYTHON_PATH"

# 실행 스크립트 경로
RUN_SCRIPT="$PROJECT_DIR/crawlers/run_daily.py"

# 실행 권한 부여
chmod +x "$RUN_SCRIPT"
echo -e "${GREEN}✓${NC} 실행 권한 부여: $RUN_SCRIPT"

# logs 디렉토리 생성
mkdir -p "$PROJECT_DIR/logs"
echo -e "${GREEN}✓${NC} 로그 디렉토리 생성: $PROJECT_DIR/logs"

# Cron job 내용
CRON_JOB="0 0 * * * cd $PROJECT_DIR && $PYTHON_PATH $RUN_SCRIPT >> $PROJECT_DIR/logs/cron.log 2>&1"

# 기존 cron job 확인
echo -e "\n${YELLOW}[1] 기존 cron job 확인${NC}"
crontab -l 2>/dev/null | grep -q "$RUN_SCRIPT"
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}⚠${NC}  이미 등록된 cron job이 있습니다."
    echo -e "${YELLOW}⚠${NC}  제거 후 다시 등록합니다."
    crontab -l 2>/dev/null | grep -v "$RUN_SCRIPT" | crontab -
fi

# Cron job 추가
echo -e "\n${YELLOW}[2] Cron job 등록${NC}"
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

# 확인
echo -e "\n${YELLOW}[3] 등록된 cron job 확인${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
crontab -l | grep "$RUN_SCRIPT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo -e "\n${GREEN}✓ Cron 설정 완료!${NC}"
echo -e "\n${YELLOW}스케줄:${NC}"
echo "  • 매일 자정 00:00에 자동 실행"
echo "  • 다나와 GPU 가격 크롤링"
echo "  • 환율 정보 수집"
echo "  • 뉴스 수집 및 감정 분석"
echo "  • 256차원 Feature 생성"

echo -e "\n${YELLOW}로그 확인:${NC}"
echo "  • Cron 로그: $PROJECT_DIR/logs/cron.log"
echo "  • 상세 로그: $PROJECT_DIR/logs/daily_crawl.log"

echo -e "\n${YELLOW}수동 실행:${NC}"
echo "  cd $PROJECT_DIR"
echo "  python3 crawlers/run_daily.py"

echo -e "\n${YELLOW}Cron 관리:${NC}"
echo "  • 목록 보기: crontab -l"
echo "  • 편집: crontab -e"
echo "  • 삭제: crontab -r"

echo -e "\n${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}설정 완료! 매일 자정에 자동으로 데이터를 수집합니다.${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
