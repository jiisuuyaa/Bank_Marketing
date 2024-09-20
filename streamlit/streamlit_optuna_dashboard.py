import streamlit as st
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import sqlite3

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, make_scorer
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna.visualization.matplotlib as optuna_viz


# 사이드바에 메뉴 추가
menu = st.sidebar.radio("메뉴를 선택하세요", ['Optuna 대시보드', '피처 선택 및 GBM 성능'])

# Optuna 대시보드 선택 시 보여줄 내용
if menu == 'Optuna 대시보드':
    # 마크다운으로 제목과 설명 꾸미기
    st.markdown("# 🎈Optuna 대시보드🎈")
    st.markdown("## 🔎가장 좋은 모델은 무엇일까낭")
    st.markdown("### Optuna를 활용하여 하이퍼파라미터 튜닝 결과를 시각화📌")
    st.markdown("---")  # 구분선

    # 4개의 열을 생성하여 이미지 배치
    col1, col2, col3, col4 = st.columns(4)

    # 이미지 URL을 통해 불러오기
    image1_url = "https://github.com/user-attachments/assets/fe421d0c-b67b-41fe-af8c-5e650e75e5d4"
    image2_url = "https://github.com/user-attachments/assets/9914a37f-9548-4e99-b1ea-0a58efce773b"
    image3_url = "https://github.com/user-attachments/assets/4b52fa99-524e-432f-816d-09de85938f04"
    image4_url = "https://github.com/user-attachments/assets/ec09a2ef-252d-446b-9bf4-849d1ad13e2c"

    # 이미지 동일한 크기로 리사이즈
    def resize_image(image_url, size=(150, 150)):
        response = requests.get(image_url)  # URL에서 이미지 다운로드
        img = Image.open(BytesIO(response.content))  # BytesIO를 사용해 Pillow로 열기
        img = img.resize(size)
        return img

    # 각 열에 이미지를 넣기
    with col1:
        st.image(resize_image(image1_url), caption='🎀하영핑')

    with col2:
        st.image(resize_image(image2_url), caption='👰🏻희성핑')

    with col3:
        st.image(resize_image(image3_url), caption='🍤상민핑')

    with col4:
        st.image(resize_image(image4_url), caption='🐰지수핑')

    # Optuna study 로드
    db_path = 'bank_marketing_optuna.db'
    engine = create_engine(f'sqlite:///{db_path}')



    try:
        study = optuna.load_study(study_name='bank_marketing_optimization', storage=f'sqlite:///{db_path}')
    except KeyError:
        st.error("해당 이름의 Study를 찾을 수 없습니다.")
        study = None

    if study:
        # 성과 기록 시각화 (Optimization History)
        st.markdown("## 최적화 기록")
        st.markdown("아래 그래프는 하이퍼파라미터 최적화 과정에서 성능이 어떻게 변했는지를 보여줍니다.")
        ax = optuna_viz.plot_optimization_history(study)
        fig = ax.get_figure()
        st.pyplot(fig)

        # 병렬 좌표 플롯 (Parallel Coordinate Plot)
        st.markdown("## 병렬 좌표 그래프")
        st.markdown("여러 하이퍼파라미터들이 모델 성능에 어떤 영향을 미쳤는지 확인할 수 있습니다.")
        ax2 = optuna_viz.plot_parallel_coordinate(study)
        fig2 = ax2.get_figure()
        st.pyplot(fig2)

        # 하이퍼파라미터 중요도 (Hyperparameter Importance)
        st.markdown("## 하이퍼파라미터 중요도")
        st.markdown("각 하이퍼파라미터가 모델 성능에 얼마나 기여했는지를 보여줍니다.")
        ax3 = optuna_viz.plot_param_importances(study)
        fig3 = ax3.get_figure()
        st.pyplot(fig3)

        from sqlalchemy import text

        # 시각화되지 않은 study 데이터프레임으로 확인
        st.markdown("## 최적화 실험 데이터")
        with engine.connect() as connection:
            result = connection.execute(text('SELECT * FROM studies'))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
        st.write(df)

        # Trials 데이터
        st.markdown("## 트라이얼 데이터")
        with engine.connect() as connection:
            result = connection.execute(text('SELECT * FROM trials'))
            df_trials = pd.DataFrame(result.fetchall(), columns=result.keys())
        st.write(df_trials)

        

    # 추가적인 정보
    st.markdown("---")  # 구분선
    st.markdown("### 제작자: 하영상민지수희성핑")
    st.markdown("[🔗 GitHub 링크](https://github.com/sangminpark9/ML041)")










# 커스텀 스코어 함수 (AUC와 Recall의 조화 평균)
def custom_score(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    recall = recall_score(y_true, y_pred_proba > 0.5, pos_label=1)
    return 2 * (auc * recall) / (auc + recall)

custom_scorer = make_scorer(custom_score, needs_proba=True)

st.markdown("# 피처 선택 및 Soft Voting 모델 성능 시각화")
st.write("CSV 파일을 업로드하고, 사용할 피처를 선택하여 Soft Voting 모델의 성능을 확인하세요.")

# CSV 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    # 데이터 로드
    df = pd.read_csv(uploaded_file)

    # 데이터프레임과 타겟 컬럼 분리
    target_column = 'deposit'  # 타겟 컬럼이 'deposit'이라고 가정
    features = df.drop(columns=[target_column, 'duration'])
    target = df[target_column]

    # 타겟 변수 변환 ('no', 'yes' -> 0, 1)
    le = LabelEncoder()
    y = le.fit_transform(target)

    # 범주형 변수를 One-Hot Encoding으로 변환
    features_encoded = pd.get_dummies(features, drop_first=True)

    # 피처 선택을 위한 멀티셀렉트
    selected_features = st.multiselect('사용할 피처를 선택하세요:', features_encoded.columns.tolist(), default=features_encoded.columns.tolist())

    # 데이터가 선택되었을 경우
    if selected_features:
        X = features_encoded[selected_features]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 모델 정의
        rf = RandomForestClassifier(random_state=42)
        xgb = XGBClassifier(random_state=42)
        lgbm = LGBMClassifier(random_state=42)

        # Voting Classifier 생성
        voting_clf = VotingClassifier([
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgbm)
        ], voting='soft', weights=[1, 1, 1])

        # 모델 학습
        voting_clf.fit(X_train, y_train)

        # 예측
        y_pred = voting_clf.predict(X_test)
        y_pred_proba = voting_clf.predict_proba(X_test)[:, 1]

        # 성능 평가
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        rec = recall_score(y_test, y_pred)

        # 성능 결과 출력
        st.write(f"**선택된 피처 수**: {len(selected_features)}")
        st.write(f"**Accuracy**: {acc:.4f}")
        st.write(f"**AUC**: {auc:.4f}")
        st.write(f"**Recall**: {rec:.4f}")

        # 추가: 각 모델의 개별 성능 비교
        st.markdown("## 개별 모델 성능 비교")
        for name, model in voting_clf.named_estimators_.items():
            y_pred_individual = model.predict(X_test)
            y_pred_proba_individual = model.predict_proba(X_test)[:, 1]
            
            acc_individual = accuracy_score(y_test, y_pred_individual)
            auc_individual = roc_auc_score(y_test, y_pred_proba_individual)
            rec_individual = recall_score(y_test, y_pred_individual)
            
            st.write(f"**{name} 모델**")
            st.write(f"Accuracy: {acc_individual:.4f}")
            st.write(f"AUC: {auc_individual:.4f}")
            st.write(f"Recall: {rec_individual:.4f}")
            st.write("---")
