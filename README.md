# 데이터 분석 스튜디오 (Streamlit)

확장 가능한 EDA·모델링·프로파일링·시각화 앱입니다. 기존 'Starter'에서 확장되어 더 풍부한 기능을 제공합니다.

## 빠른 시작

1) 가상환경 생성 및 패키지 설치

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2) 앱 실행

```
streamlit run app.py
```

3) 사용 방법
- 좌측 사이드바에서 CSV 업로드 또는 샘플 데이터 사용 선택
- 상단 탭에서 개요/탐색/시각화/변환(내보내기)/모델(Stub) 기능 사용

## 포함 기능
- CSV 업로드(구분자/인코딩/소수점 설정)
- 샘플 데이터 생성(행 수/시드)
- 개요: 미리보기, 결측치 요약, 상관관계 히트맵
- 탐색: 수치형 요약, 범주별 분포
- 시각화: 히스토그램/산점도/박스플롯/막대(빈도)/선형(시계열)
  - 고급: 산점도 행렬, 2D 밀도(컨투어/히트맵), 페이싯 히스토그램/바이올린
- 변환: 컬럼 선택, 결측치 제거, Query 필터, 샘플링, CSV 다운로드
- 모델: 분류/회귀 베이스라인 + 하이퍼파라미터 튜닝(Grid/Random + CV),
  지표(Accuracy/F1, MAE/RMSE/R2), 혼동행렬/예측-실측/잔차 히스토그램,
  CV 결과표/최적 파라미터/Best CV Score 표시
 - 해석: 피처 중요도(트리) / 계수(선형/로지스틱) 시각화
 - 예측/내보내기: 학습된 모델로 전체/업로드 데이터에 대한 예측 생성,
   확률 포함 옵션(분류) 및 CSV 다운로드
 - 전처리(고급): 모델 탭에서 결측 전략(수치/범주), 스케일링(Standard/MinMax/Robust),
   OneHot drop 옵션(first/if_binary) 선택 및 IQR 이상치 제거(학습 전) 지원
 - 변환 탭: 중복 제거(부분/전체, first/last/모두 제거), IQR 이상치 제거 지원

## 구조
```
Data_Analysis_Project/
├─ app.py                # Streamlit 메인 앱
├─ requirements.txt      # 의존성 목록
├─ README.md             # 사용 안내
└─ src/
   ├─ data_ops.py        # 데이터 로드/샘플/타입 식별
   ├─ eda_ops.py         # 요약/결측/상관 계산
   └─ viz_ops.py         # Plotly 시각화 유틸
```

## 확장 아이디어
- 데이터 전처리 파이프라인(스케일링/인코딩/피처선택) 탭 추가
 - 예측/내보내기: 학습된 모델로 예측 컬럼 추가 후 CSV 다운로드
- 프로파일링(예: ydata-profiling, sweetviz) 통합
- 앱 테마/레이아웃 사용자화 및 다국어 지원
## 앱 이름(리포지토리/폴더명) 반영
- 기본적으로 `src/settings.py`가 현재 프로젝트 폴더명을 Title Case로 변환하여 앱 이름(`APP_NAME`)으로 사용합니다.
- `run_app.bat`는 폴더명에서 하이픈/언더스코어를 공백으로 치환해 `APP_NAME` 환경변수를 설정합니다.
- 고정하고 싶다면 실행 전 환경변수로 지정하세요:
  - PowerShell/CMD: `set APP_NAME=내 앱 이름 & run_app.bat`
  - Bash: `APP_NAME="내 앱 이름" streamlit run app.py`
- 태그라인은 `APP_TAGLINE` 환경변수 또는 `src/settings.py`의 기본값으로 제어합니다.
