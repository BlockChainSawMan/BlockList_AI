# start.sh 파일 내용

#!/bin/bash
# 1. 파일 다운로드 (800MB)
python download_files_only.py

# 2. 다운로드 성공 시 FastAPI 서버 시작(Uvicorn)실행 ($PORT 변수 사용)
uvicorn api_server:app --host 0.0.0.0 --port $PORT