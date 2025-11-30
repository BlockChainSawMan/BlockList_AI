# 1. 최소 이미지
FROM python:3.10-slim

# 2. 작업 디렉토리
WORKDIR /app

# 3. 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# 5. 소스 코드 복사
COPY . .

# 6. Railway에서 사용되는 PORT 기본값 지정
ENV PORT=8000

# 7. ENTRYPOINT

# 실행권한 추가
RUN chmod +x /app/start.sh

# ENTRYPOINT는 start.sh "만" 실행한다
ENTRYPOINT ["bash", "/app/start.sh"]