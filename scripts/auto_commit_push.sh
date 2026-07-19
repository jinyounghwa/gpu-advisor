#!/bin/bash
# 매일 새벽 1시 자동 커밋 & 푸시 (LaunchAgent: com.gpu-advisor.auto-commit)
# 자정 크롤링(run_daily.py)이 생성한 데이터/위키 변경사항을 커밋하고 origin/main에 푸시한다.
# 로그: ~/Library/Logs/gpu-advisor/auto-commit.log (launchd stdout/stderr)

set -u

REPO="/Users/younghwa.jin/Documents/gpu-advisor"
BRANCH="main"
export GIT_TERMINAL_PROMPT=0   # 인증 실패 시 프롬프트 대기 없이 즉시 실패
export PATH="/usr/local/bin:/usr/bin:/bin"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

cd "$REPO" || { log "ERROR: 저장소 디렉토리 이동 실패: $REPO"; exit 1; }

# 크롤링(자정 시작)이 아직 실행 중이면 인덱스 충돌 방지를 위해 대기 (최대 30분)
for i in $(seq 1 30); do
    pgrep -f "crawlers/run_daily.py" > /dev/null || break
    log "run_daily.py 실행 중... 60초 대기 ($i/30)"
    sleep 60
done

if pgrep -f "crawlers/run_daily.py" > /dev/null; then
    log "ERROR: 30분 대기 후에도 run_daily.py가 실행 중. 오늘 커밋 건너뜀."
    exit 1
fi

if [ -z "$(git status --porcelain)" ]; then
    log "변경사항 없음. 커밋 건너뜀."
    exit 0
fi

git add -A
git commit -m "docs: 일일 데이터 및 위키 업데이트 ($(date '+%Y-%m-%d')) [auto]" \
    || { log "ERROR: 커밋 실패"; exit 1; }
log "커밋 완료: $(git log --oneline -1)"

# 푸시 (실패 시 5분 간격 3회 재시도 - 일시적 네트워크 문제 대비)
for attempt in 1 2 3; do
    if git push origin "$BRANCH" 2>&1; then
        log "푸시 성공 (시도 $attempt)"
        exit 0
    fi
    log "푸시 실패 (시도 $attempt/3). 300초 후 재시도..."
    [ "$attempt" -lt 3 ] && sleep 300
done

log "ERROR: 푸시 3회 실패. 커밋은 로컬에 보존됨. 다음 실행 때 함께 푸시됨."
exit 1
