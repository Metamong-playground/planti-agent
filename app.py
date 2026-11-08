import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os
import time

# 1. 페이지 기본 설정
st.set_page_config(page_title="Planti AI Terminal", page_icon="🌱", layout="wide")

st.title("🌱 Planti (플랜티) - AI 모니터링 시스템")
st.caption("컴퓨터 비전 & Scikit-learn 기반 실시간 식물 진단 엔진")

# 2. AI 모델 로드
@st.cache_resource
def load_model():
    if os.path.exists("plant_model.pkl"):
        return joblib.load("plant_model.pkl")
    return None

model = load_model()

# OpenCV 특징 추출 함수 (실시간 터미널 로깅 기능 포함)
def extract_features_with_log(img_pil, log_container):
    logs = []
    
    def add_log(msg):
        timestamp = time.strftime("[%H:%M:%S]")
        logs.append(f"{timestamp} {msg}")
        log_container.code("\n".join(logs), language="bash")
        time.sleep(0.3) # 터미널처럼 출력되는 느낌을 주기 위한 미세 대기

    add_log("Initializing OpenCV Image Processing Pipeline...")
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    add_log(f"Resizing input tensor to (256, 256, 3)...")
    img = cv2.resize(img, (256, 256))
    
    add_log("Converting Color Space: BGR -> HSV...")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. 녹색 (healthy)
    add_log("Extracting Feature [1/4]: Green Color Ratio (Healthy Index)...")
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / (256 * 256)
    
    # 2. 반점 (disease)
    add_log("Extracting Feature [2/4]: Brown Spot Density (Disease Index)...")
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    spot_ratio = np.sum(cv2.inRange(hsv, lower_brown, upper_brown) > 0) / (256 * 256)
    
    # 3. 황화 (sunlight)
    add_log("Extracting Feature [3/4]: Yellow Leaf Index (Sunlight Deficiency)...")
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / (256 * 256)
    
    # 4. 시듦 (water)
    add_log("Extracting Feature [4/4]: Edge Browning Ratio (Water Stress)...")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    water_need = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        mask_edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.drawContours(mask_edge, [cnt], -1, 255, thickness=15)
        edge_brown = cv2.bitwise_and(cv2.inRange(hsv, lower_brown, upper_brown), mask_edge)
        water_need = np.sum(edge_brown > 0) / (np.sum(mask_edge > 0) + 1e-5)

    add_log(f"Feature Vector Extracted: [{g_ratio:.3f}, {spot_ratio:.3f}, {sunlight_index:.3f}, {water_need:.3f}]")
    add_log("Feeding Feature Vector into 'plant_model.pkl' (RandomForest)...")
    
    return [g_ratio, spot_ratio, sunlight_index, water_need]

# 결과 진단 및 분석 함수
def predict_and_display(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="입력된 식물 이미지", use_container_width=True)
    
    with col2:
        st.subheader("🖥️ AI 실시간 처리 콘솔 (Terminal Output)")
        log_box = st.empty() # 터미널 로그가 찍힐 창
        
        if model is None:
            st.error("🚨 plant_model.pkl 파일이 없습니다!")
            return

        # 실시간 진행 콘솔 출력
        features = extract_features_with_log(image, log_box)
        prediction = model.predict([features])[0]

        st.success("✅ AI 분류 작업 완료!")

    st.divider()
    st.markdown("### 📊 최종 진단 결과 및 리포트")
    
    if prediction == 'healthy':
        st.success("✅ **상태: 건강함 (Healthy)**")
        st.info("💡 식물이 튼튼하게 자라고 있습니다! 현재 수분과 일조량을 유지해 주세요.")
    elif prediction == 'disease':
        st.error("⚠️ **상태: 병충해 의심 (Disease)**")
        st.warning("💡 잎 반점이 검출되었습니다. 잎 뒷면을 확인하고 살균제를 처방해 주세요.")
    elif prediction == 'sunlight':
        st.warning("☀️ **상태: 일조량 부족 (Sunlight Needed)**")
        st.info("💡 잎이 황화 현상을 보이고 있습니다. 창가 쪽으로 옮겨주세요.")
    elif prediction == 'water':
        st.warning("💧 **상태: 수분 부족 (Water Needed)**")
        st.info("💡 잎 테두리 시듦이 감지되었습니다. 흙 상태를 확인하고 물을 듬뿍 주세요.")

# 3. 메인 인터페이스
st.markdown("### 📸 식물 이미지 입력")
tab1, tab2 = st.tabs(["📷 실시간 촬영", "📁 이미지 업로드"])

with tab1:
    camera_image = st.camera_input("카메라 촬영")
    if camera_image:
        image = Image.open(camera_image)
        predict_and_display(image)

with tab2:
    uploaded_file = st.file_uploader("파일 첨부", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        predict_and_display(image)
