import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Planti AI", layout="wide")

# 연두색 소프트 파스텔 배경 커스텀 CSS 적용
st.markdown(
    """
    <style>
    .stApp {
        background-color: #E8F5E9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Planti - 식물 상태 진단 에이전트")
st.caption("식물 건강을 간편하게 확인")

# 0. 진단 히스토리 세션 상태 초기화
if "history" not in st.session_state:
    st.session_state.history = []

# 1. AI 모델 로드 (예외 처리 및 백업)
@st.cache_resource
def load_model():
    model_path = "plant_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    
    # 백업 룰-베이스 머신러닝 분류기
    X_dummy = [
        [0.6, 1.0, 0.1, 0.0],  # healthy
        [0.1, 8.0, 0.1, 0.1],  # disease
        [0.2, 1.0, 0.6, 0.0],  # sunlight
        [0.2, 1.0, 0.0, 0.5]   # water
    ]
    y_dummy = ['healthy', 'disease', 'sunlight', 'water']
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_dummy, y_dummy)
    return clf

model = load_model()

# 2. 이미지 면적 비례 동적 반점 및 생리학 지표 추출 함수
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    total_pixels = 256 * 256  # 65,536 px
    
    # 잎 면적 대비 점(Spot) 검출 동적 픽셀 범위 (Bousset et al., 2016)
    min_spot_area = total_pixels * 0.0001  # 약 6 px
    max_spot_area = total_pixels * 0.0050  # 약 320 px
    
    # 1) 녹색 비율 (GLAI)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / total_pixels
    
    # 2) 갈색/검은색 반점 덩어리 추출
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    blurred = cv2.GaussianBlur(brown_mask, (5, 5), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spot_count = 0
    large_patch_area = 0.0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_spot_area <= area <= max_spot_area:
            spot_count += 1
        elif area > max_spot_area:
            large_patch_area += area
            
    large_patch_ratio = large_patch_area / total_pixels
    
    # 3) 황화 현상 (Zhu et al., 2024)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / total_pixels
    
    # 4) 시듦 및 테두리 괴사 현상 (Nevin & Lovatt, 1987)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    edge_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    water_need = 0.0
    if edge_contours:
        cnt = max(edge_contours, key=cv2.contourArea)
        mask_edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.drawContours(mask_edge, [cnt], -1, 255, thickness=20)
        edge_brown = cv2.bitwise_and(brown_mask, mask_edge)
        water_need = np.sum(edge_brown > 0) / (np.sum(mask_edge > 0) + 1e-5)

    return [g_ratio, float(spot_count), sunlight_index, water_need, large_patch_ratio]

# 3. 진단 및 모니터링 출력 모듈
def predict_and_display(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="
