#download_utils.py
import gdown
import os
from pathlib import Path

# Google Drive 파일 ID 추출 및 다운로드 URL 생성
FILE_URLS = {
    "data/df_merged.csv": "https://drive.google.com/uc?id=1wgqPFQU59k6ClHpnp2p2I5eVIdKwJJ6F",
    "data/elliptic_node_final.csv": "https://drive.google.com/file/d/1cyu2sXKnKJ0OzSX2LHF1SrbyHpKcWcYc",
    "data/elliptic_data_v2.pt": "https://drive.google.com/uc?id=1nrXtCC06hrMvYR7pHUhpcFf34yU2MWwp",
    "models/saved/elliptic_gat_best.pt": "https://drive.google.com/uc?id=1-nhrMHHAIaKsSFU0SPxrBg9iiuUI54WX",
    "models/saved/explainer_pg.pt": "https://drive.google.com/uc?id=1Fv3D37RL6eVEB0WdyVuDgtOM3NVxZVgY"
}

def download_files():
    """서버 시작 시 필요한 파일들을 Google Drive에서 다운로드"""
    for filepath, url in FILE_URLS.items():
        path = Path(filepath)
        
        # 파일이 이미 존재하면 건너뛰기
        if path.exists():
            print(f"✓ {filepath} already exists")
            continue
        
        # 디렉토리 생성
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 다운로드
        print(f"Downloading {filepath}...")
        try:
            gdown.download(url, str(path), quiet=False)
            print(f"✓ {filepath} downloaded successfully")
        except Exception as e:
            print(f"✗ Failed to download {filepath}: {e}")
            raise

if __name__ == "__main__":
    download_files()