import streamlit as st
import optuna
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import optuna.visualization.matplotlib as optuna_viz
from PIL import Image
import requests
from io import BytesIO

# 마크다운으로 제목과 설명 꾸미`기
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

# SQLite 데이터베이스에서 데이터 로드
db_path = 'C:/Users/jiisuu/Desktop/forder/ML041/bank_marketing_optuna.db'
engine = create_engine(f'sqlite:///{db_path}')

# Optuna study 로드
study = optuna.load_study(study_name='bank_marketing_optimization', storage=f'sqlite:///{db_path}')

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

# 시각화되지 않은 study 데이터프레임으로 확인
st.markdown("## 최적화 실험 데이터")
st.markdown("Optuna로 수행한 실험 데이터를 아래에서 확인할 수 있습니다.")
df = pd.read_sql('SELECT * FROM studies', engine)
st.write(df)

# Trials 데이터
st.markdown("## 트라이얼 데이터")
st.markdown("각 트라이얼의 하이퍼파라미터 조합과 성능을 확인할 수 있습니다.")
df_trials = pd.read_sql('SELECT * FROM trials', engine)
st.write(df_trials)

# 추가적인 정보
st.markdown("---")  # 구분선
st.markdown("### 제작자: 하영상민지수희성핑")
st.markdown("[🔗 GitHub 링크](https://github.com/sangminpark9/ML041)")

# 구글 폰트 '나눔고딕' 적용
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');

    * {
        font-family: 'Noto Sans KR', sans-serif;
    }
    h1 {
        color: #ff3333;  /* 빨강 */
        text-align: center;
    }
    h2, h3 {
        color: #000000;  /* 검정 */
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)
