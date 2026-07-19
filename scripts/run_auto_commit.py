#!/usr/local/bin/python3
"""LaunchAgent 진입점: auto_commit_push.sh 실행 래퍼.

launchd가 /bin/bash를 직접 실행하면 macOS TCC가 ~/Documents/ 접근을 차단하므로
(Operation not permitted, exit 126), Full Disk Access가 부여된 python3를
진입점으로 사용하고 여기서 bash 스크립트를 subprocess로 실행한다.
(com.gpu-advisor.daily-crawl과 동일한 패턴)
"""
import subprocess
import sys

SCRIPT = "/Users/younghwa.jin/Documents/gpu-advisor/scripts/auto_commit_push.sh"

sys.exit(subprocess.call(["/bin/bash", SCRIPT]))
