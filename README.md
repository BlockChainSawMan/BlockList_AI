# Blockchainsawman_Blocklist
Blocklist : 온체인 기반 스테이블 코인 거래 AML 서비스

## 🧑‍💻 개발 환경 및 기술 스택

> **Language**

<img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white"> <img src="https://img.shields.io/badge/Cypher-008CC1?style=flat-square&logo=neo4j&logoColor=white">

> **Tool**

<img src="https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=flat-square&logo=visual-studio-code&logoColor=white"> <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white">

> **Stack**

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"> <img src="https://img.shields.io/badge/PyTorch%20Geometric-3793EF?style=flat-square&logo=pyg&logoColor=white"> <img src="https://img.shields.io/badge/LangChain-1A73E8?style=flat-square&logo=chainlink&logoColor=white"> <img src="https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white"> <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white">

> **Database**

<img src="https://img.shields.io/badge/Neo4j-008CC1?style=flat-square&logo=neo4j&logoColor=white">



## 🚧 설계

### 📉 시스템 아키텍처

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


## 📂 AI Part

### 0. Dataset

[Elliptic Dataset](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)

Elliptic 데이터셋은 비트코인 거래 데이터를 포함하고 합법 및 불법 엔티티(사기, 멀웨어, 테러 조직, 랜섬웨어, 폰지 사기 등)가 매핑되어 있습니다. 노드와 엣지 데이터셋으로 분류되어 있어 Neo4j 지식그래프 구축에 적합한 형태이며, 본 프로젝트에서는 해당 비트코인 거래 데이터셋을 스테이블 코인 거래 데이터셋으로 가정 후 진행하였습니다.

### 1. GNN-Based Anomaly Detection: Final Methodology & Evolution

![unnamed](https://github.com/user-attachments/assets/637c5f5d-c08f-44a8-98cb-000ce5b2f891)

### 2. RAG
