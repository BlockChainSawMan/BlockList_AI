# Blockchainsawman_Blocklist
Blocklist : 온체인 기반 스테이블 코인 거래 AML 서비스

## 🧑‍💻 개발 환경 및 기술 스택

### AI & Data Analysis

[Python] [PyTorch] [PyTorch Geometric] [Pandas] [NumPy]



## 🚧 설계

### 시스템 아키텍처

[AI 모델 학습 및 서빙 파이프라인을 포함한 시스템 아키텍처 다이어그램 이미지를 삽입예정 : 데이터 전처리 -\> GNN 모델 학습 -\> 모델 저장 -\> FastAPI를 통한 모델 서빙 및 OpenAI API 연동]

### 📂 AI Part Directory

```bash
.
├── data                  # AI 모델 학습 및 분석을 위한 데이터 디렉토리
│   ├── df_merged.csv     # 병합된 노드 데이터셋 (unknown 라벨 포함)
│   ├── elliptic_data_v2.pt # PyTorch Geometric용 전체 그래프 데이터 객체 파일
│   └── elliptic_node_final.csv # 최종 전처리된 노드 피처 데이터셋
├── models                # 학습된 모델 저장소
│   └── saved
│       ├── elliptic_gat_best.pt # 최적의 성능을 낸 GNN GAT 학습 모델 파일
│       ├── explainer_pg.pt      # GNN 예측 결과 해석을 위한 Explainer 모델 파일
├── .env                  # 환경 변수 설정 파일 (Neo4j 접속 정보, OpenAI API Key 등)
├── .gitignore            # Git 버전 관리 제외 파일 목록
├── api_server.py         # FastAPI 기반 AI 모델 서빙 서버 코드
├── final.py              # 데이터 로드, 모델 추론, 결과 분석 등 전체 분석 로직 실행 코드
├── README.md             # 프로젝트 설명 문서
└── requirements.txt      # 필요한 파이썬 라이브러리 및 버전 목록
```


# GNN-Based Anomaly Detection: Final Methodology & Evolution

![unnamed](https://github.com/user-attachments/assets/637c5f5d-c08f-44a8-98cb-000ce5b2f891)
