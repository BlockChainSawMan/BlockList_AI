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
├── data                  # AI 모델학습을 위한 데이터 셋
│   ├── df_merged.csv     # 병합된 노드 데이터셋 (unknown 라벨 포함)
│   ├── elliptic_data_v2.pt # PyTorch Geometric용 전체 그래프 데이터 객체 파일
│   └── elliptic_node_final.csv # 최종 전처리된 노드 피처 데이터셋
│
├── model_train           # AI 모델학습 과정
│   └── train_gnn+pge_explainer.ipynb # GNN GAT + Explainer 모델 학습 파일
│
├── models                # 학습된 모델 저장소
│   └── saved
│       ├── elliptic_gat_best.pt # 최적의 성능을 낸 GNN GAT 학습 모델 파일
│       └── explainer_pg.pt      # GNN 예측 결과 해석을 위한 Explainer 모델 파일
│
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

<img width="700" height="600" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/e3d85678-acdc-4aa5-b645-9f05abaceab5" />

Elliptic 데이터셋은 비트코인 거래 데이터를 포함하고 `elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv` 3개의 파일로 구성되어 있습니다. 노드와 엣지 기반으로 분류되어 있어 Neo4j 지식그래프 구축에 적합한 형태입니다.

① 거래 관계도 (elliptic_txs_edgelist.csv) 
- 행: 203,769개 열: 2개
- 거래 관계 즉, 자금 이동 경로를 나타내는 데이터
-  A지갑에서 B지갑으로 코인이 이동했다면, 이는 '거래(Transaction)'라는 노드 간의 연결로 표현됨

 | txId1 (출발 노드) | txId2 (도착 노드) | 
 | :--- | :--- | 
 | 230425980 | 5530458 | 

② 거래 속성 정보 (elliptic_txs_features.csv) 
- 행: 203,769개 트랜잭션 (노드) 열: 167개
- 각 거래(노드)가 가지는 166차원의 고유한 특징 벡터로 해당 거래 자체의 특성(Local Features)와 이를 기반으로 한 집계 정보(Aggregated Features)로 이루어짐
- 해당 데이터셋에서는 개인정보보안을 위해 features의 특징이 익명화되어 있음

```
1. Local Features (93개)
- 해당 거래 자체의 정보
- (예: 타임스탬프 외 수수료, 입/출금 횟수, 거래량 등 정보 익명화 )
2. Aggregated Features (73개)
- 해당 거래와 연결된 이웃 노드들의 거래 기반 통계 정보
- (예: 이웃 거래들의 평균 거래량, 이웃의 이웃이 가진 표준편차 등)
```
③ 정답 레이블 (elliptic_txs_classes.csv) 
- 행: 203,769개 열: 2개
- 거래 정상(0)/비정상(1)/미분류(-1)를 나타내는 데이터, 불균형 데이터의 특징
- 데이터 분포
  - 불법 (illicit) - 약 4,545개 (2%)
  - 합법 (licit) - 약 42,019개 (21%)
  - unknown = 미분류 - 약 157,205개 (77%)
```
- 0 (Licit): 거래소, 지갑 서비스 등 합법적 거래 (정상)
- 1 (Illicit): 다크웹, 랜섬웨어, 자금세탁 등 불법 거래 (비정상)
- Unknown: 아직 분류되지 않은 거래 (전체의 약 98% 차지, 준지도 학습 활용 가능)
```

### 1. GNN-Based Anomaly Detection : Final Methodology & Evolution


<img width="2848" height="1504" alt="gnnmodel_info" src="https://github.com/user-attachments/assets/6c5610e2-f5b1-471c-bb51-affce8ddcf3e" />


GNN모델 학습 단계에서는 GAT 모델 도입 초기에 데이터 정규화 부재와 클래스 불균형으로 인한 Gradient Explosion(NaN 발생) 및 낮은 F1-Score(0.04) 문제에 직면했습니다. 이후 GCN 변경과 Weighted Loss, BatchNorm 등을 도입하여 학습 안정성과 Recall을 개선했으나, 모델이 그래프 구조(Edge)보다 노드 자체 피처에 과도하게 의존하여 단순 지도학습과 다를 바 없는 결과(정확도 0.9799)를 보였습니다. 이에 EdgeForcedGATNet을 최종 고안하여 Input Feature와 Edge에 과감한 Dropout을 적용하여 노드 피처 의존도를 낮추고 이웃 정보 학습을 강제했습니다. 결과적으로 단순 정확도는 조정되었으나(0.8878), 엣지 정보 활용이 검증된 견고한 그래프 학습 모델을 구축할 수 있었습니다.

XAI 모델의 'Feature Dominance' 문제와 해결과정은 다음과 같습니다. 학습된 모델을 설명하기 위해 초기엔 GNN모델의 거래 관계의 중요한 Node정보를 중요도 순으로 출력하는 GNNExplainer를 학습시켰으나, 실시간 금융 사기 탐지 시스템의 특성상 Low Latency(낮은지연시간)가 필수적이라 판단하여 Inductive 방식의 PGExplainer로 고도화하였습니다. 이를 통해 설명 생성 시간을 수 초(sec) 단위에서 0.0X초(ms) 단위로 단축했습니다. PGExplainer 학습 과정에서 모든 엣지의 중요도(Mask)가 0.0000으로 수렴하는 현상 발생을 해결하기 위해 Learning Rate를 0.003에서서 0.0005 수준으로 낮춰 안정성을 확보하고, 규제 계수를 완화하여 로그 연산의 영향력을 줄여 개선하였습니다. 마지막으로 데이터 셋에서 최다수를 차지하는 미분류 거래의 영향력으로 NaN값 전파되는 현상을 방지하기 위해 NaN 발생 시 해당 배치는 무시하여 Loop 내에서 total_loss가 오염되지 않도록 학습했습니다.

[학습 환경]
- **GPU** : NVIDIA A100-SXM4-40GB(Google Colab 환경)
- **Framework** : PyTorch 2.9.0, PyTorch Geometric
- **데이터 분할** : Train 60% / Val 20% / Test 20%


### 2. Neo4j Knowledge Graph Construction & Evidence Retrieval Pipeline

<img width="2816" height="1504" alt="Gemini_Generated_Image_nyhirknyhirknyhi" src="https://github.com/user-attachments/assets/169efb7d-3fc6-49f8-aed0-a940aa887230" />

GNN 모델&PGExplainer를 통해 특정 거래패턴의 비정상적인 움직임을 주변 노드정보를 기반으로 전달받은 후, Neo4j 지식그래프를 구축하고 RAG LLM 기반 설명을 통해 AI가 다음과 같은 추론을 출력하는 시스템을 구축하였습니다.
- `ID 99201 거래는 자체 속성(금액, 시간)은 정상이지만, 2단계 건너편(2-hop)에 있는 노드들이 최근 불법 자금(Class 1)과 다수 연결되어 있으므로, 이 거래 또한 '자금 세탁의 중간 경로'일 확률이 85%이다`

먼저, 데이터 전처리 단계에서는 Elliptic 데이터셋의 피처(feature)와 클래스(class) 파일을 병합하고, 분석에 불필요한 'unknown' 클래스 데이터를 제거하여 정확도를 높였습니다. 또한, 지식 그래프 내에서 엣지(Edge)의 속성을 풍부하게 만들기 위해, 기존 엣지 리스트에 존재하지 않던 '거래 금액(amount)'과 '시간(time)' 정보를 랜덤으로 생성하여 추가함으로써, 최종적으로 약 4만 6천 개의 노드와 3만 6천 개의 엣지로 구성된 정제된 데이터셋을 마련하였습니다.

이어지는 추가 변수 매핑 단계에서는 향후 연동될 거대언어모델(LLM)이 그래프의 맥락을 더 잘 이해할 수 있도록 노드 정보를 강화하였습니다. 불법 거래(Class 1)로 식별된 노드에는 'Hacker', 'Ransomware', 'Lazarus Group'과 같은 구체적인 entity_type과 commet를 부여하고, 합법 거래(Class 2)에는 'Normal User' 정보를 매핑하여 데이터의 의미론적 가치를 높였습니다.

준비된 데이터는 Neo4j 지식그래프 구축 단계를 통해 실제 데이터베이스로 이관되었습니다. 노드와 엣지 적재 함수를 각각 구현하여 정제된 데이터프레임을 Neo4j Aura DB에 적재함으로써, 거래 관계를 시각화하고 질의할 수 있는 환경을 조성하였습니다.

마지막으로 조회 로직 구축 단계에서는 특정 지갑 주소를 입력받아 연관된 거래 증거를 추출하는 파이프라인을 완성하였습니다. 사용자가 단일 또는 다수의 지갑 주소를 입력하면, 시스템은 자동으로 Cypher 쿼리를 생성하여 해당 지갑과 1촌 관계에 있는 노드 및 엣지 정보를 서브그래프(Evidence Graph) 형태로 조회합니다. 이 결과는 최종적으로 JSON 형식으로 출력되어, LLM이 답변을 생성하는데 구체적인 증거 데이터로 활용될 수 있도록 구현되었습니다.

### 3. LangChain-based LLM Analysis & Automated AML Reporting Pipeline

<img width="2848" height="1504" alt="Gemini_Generated_Image_6wtou96wtou96wto" src="https://github.com/user-attachments/assets/e18d1dca-2579-4623-9105-483cb643950d" />

이 단계는 랭체인이 앞서 구축된 GNN GAT 모델의 결과값과 Neo4j 조회 파이프라인에서 출력된 1촌 서브그래프를 참조해 RAG 기반 검색으로 LLM 자연어 답변 생성을 구현합니다.

이 단계에서는 랭체인 프레임워크와 GPT-4o 모델을 활용합니다. 시스템을 '블록체인 AML 분석 전문가' 페르소나로 설정하고, 앞서 전달받은 서브그래프(Evidence Graph)와 위험 점수(risk score)를 프롬프트에 주입합니다. 이때, Pydantic 파서를 적용하여 LLM의 답변이 단순 텍스트가 아닌 '상세 설명', '요약(20자 이하)', '핵심 불렛 포인트(3개)'의 정해진 JSON 규격을 엄격히 따르도록 제어합니다.

마지막으로 전체 파이프라인 통합 단계에서는 run_pipeline 함수를 통해 이 모든 과정을 하나로 묶습니다. [GNN, PGExplainer 탐지 → Neo4j 조회 → LLM 분석]의 흐름이 한 번의 실행으로 이루어지며, 최종적으로 사용자는 시각화 가능한 그래프 데이터와 함께, 해당 거래가 왜 위험한지 설명하는 자연어 설명을 동시에 받아볼 수 있게 됩니다.

### 4. Railway API

최종적으로 API 파이프라인 구성 및 실행하였습니다.    
[배포 환경]
- Dockerfile을 사용한 컨테이너 기반 환경에서 빌드 및 실행
- FastAPI 기반 백엔드 서버 코드 구성
1. 데이터 준비 단계에서 download_files_only.py 스크립트가 실행 단계에서 `gdown`을 활용하여 800MB에 달하는 대용량 데이터 및 모델 파일을 Google Drive에서 다운로드합니다. 이 과정은 빌드 시간 초과(10분)를 회피하기 위해 Procfile (혹은 start.sh)에 의해 서버 실행 명령 이전에 분리되어 실행됩니다. 다음으로 모델 로딩 단계에서 다운로드된 파일들은 api_server.py가 시작될 때 CPU 버전의 PyTorch를 사용하여 메모리에 로드됩니다.
2. FastAPI 프레임워크를 기반으로 서버가 구축되어 높은 성능의 비동기 처리를 지원합니다 `/v1/check_wallet` 엔드포인트가 정의되어 있으며, 이는 지갑 주소 또는 트랜잭션 $\text{ID}$ 목록을 받아 분석을 요청합니다.
3. Pydantic 모델인 WalletRequest를 사용하여 요청 본문(txIds: list[str])의 데이터 유효성 검사를 자동으로 수행합니다.
4. 최종적으로, API의 핵심 로직인 check_wallet 함수 내부에서 호출되는 run_pipeline 함수가 호출되어 전달받은 트랜잭션ID를 기반으로 로드된 GNN 모델 및 데이터를 활용하여 위험 탐지 분석을 실행하여 [데이터 매핑 → 모델 추론 → 분석 결과 반환]의 전체 과정이 통합되어, 사용자가 요청한 트랜잭션 ID에 대한 위험 점수 및 분석 결과를 JSON 형태로 반환합니다.
