# Google Drive에서 파일 다운로드
from download_utils import download_files  

print("Checking and downloading required files...")
download_files()
print("All files ready!")


# Dockerfile only (ENTRYPOINT)
#    ↓
# bash -c "download_files → uvicorn 실행"
#    ↓
# api_server.py에서 모델 로딩
#    ↓
# Railway 자동 포트 주입($PORT)
#    ↓
# 서비스 정상 작동
# * start.sh는 선택이 아니라, ‘예측 가능한 동작’을 위해 유지