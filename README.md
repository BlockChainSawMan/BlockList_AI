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

Elliptic 데이터셋은 비트코인 거래 데이터를 포함하고 elliptic_txs_features.csv, elliptic_txs_classes.csv, elliptic_txs_edgelist.csv 파일로 구성되어 있습니다. 노드와 엣지 기반으로 분류되어 있어 Neo4j 지식그래프 구축에 적합한 형태이며, 본 프로젝트에서는 해당 비트코인 거래 데이터셋을 스테이블 코인 거래 데이터셋으로 가정 후 진행하였습니다.


### 1. GNN-Based Anomaly Detection: Final Methodology & Evolution

![unnamed](https://github.com/user-attachments/assets/637c5f5d-c08f-44a8-98cb-000ce5b2f891)

### 2. Neo4j Knowledge Graph Construction & Evidence Retrieval Pipeline

<img width="2816" height="1504" alt="Gemini_Generated_Image_nyhirknyhirknyhi" src="https://github.com/user-attachments/assets/169efb7d-3fc6-49f8-aed0-a940aa887230" />

데이터 전처리 단계에서는 Elliptic 데이터셋의 피처(feature)와 클래스(class) 파일을 병합하고, 분석에 불필요한 'unknown' 클래스 데이터를 제거하여 정확도를 높였습니다. 또한, 지식 그래프 내에서 엣지(Edge)의 속성을 풍부하게 만들기 위해, 기존 엣지 리스트에 존재하지 않던 '거래 금액(amount)'과 '시간(time)' 정보를 랜덤으로 생성하여 추가함으로써, 최종적으로 약 4만 6천 개의 노드와 3만 6천 개의 엣지로 구성된 정제된 데이터셋을 마련하였습니다.

이어지는 추가 변수 매핑 단계에서는 향후 연동될 거대언어모델(LLM)이 그래프의 맥락을 더 잘 이해할 수 있도록 노드 정보를 강화하였습니다. 불법 거래(Class 1)로 식별된 노드에는 'Hacker', 'Ransomware', 'Lazarus Group'과 같은 구체적인 entity_type과 commet를 부여하고, 합법 거래(Class 2)에는 'Normal User' 정보를 매핑하여 데이터의 의미론적 가치를 높였습니다.

준비된 데이터는 Neo4j 지식그래프 구축 단계를 통해 실제 데이터베이스로 이관되었습니다. 노드와 엣지 적재 함수를 각각 구현하여 정제된 데이터프레임을 Neo4j Aura DB에 적재함으로써, 거래 관계를 시각화하고 질의할 수 있는 환경을 조성하였습니다.

마지막으로 조회 로직 구축 단계에서는 특정 지갑 주소를 입력받아 연관된 거래 증거를 추출하는 파이프라인을 완성하였습니다. 사용자가 단일 또는 다수의 지갑 주소를 입력하면, 시스템은 자동으로 Cypher 쿼리를 생성하여 해당 지갑과 1촌 관계에 있는 노드 및 엣지 정보를 서브그래프(Evidence Graph) 형태로 조회합니다. 이 결과는 최종적으로 JSON 형식으로 출력되어, LLM이 답변을 생성하는데 구체적인 증거 데이터로 활용될 수 있도록 구현되었습니다.
