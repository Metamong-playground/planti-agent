import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os

# 1. 페이지 기본 설정
st.set_page_config(page_title="Planti", page_icon="🌱")

st.title("🌱 Planti (플랜티)")
st.subheader("AI 기반 식물 진단 & 모니터링")

# 2. AI 모델 로드
@st.cache_resource
def load_model():
    if os.path.exists("plant_model.pkl"):
        return joblib.load("plant_model.pkl")
    return None

model = load_model()

# OpenCV 특징 추출 함수
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. 녹색 (healthy)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / (256 * 256)
    
    # 2. 반점 (disease)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    spot_ratio = np.sum(cv2.inRange(hsv, lower_brown, upper_brown) > 0) / (256 * 256)
    
    # 3. 황화 (sunlight)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / (256 * 256)
    
    # 4. 시듦 (water)
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

    return [g_ratio, spot_ratio, sunlight_index, water_need]

# 결과 진단 및 처방 출력 함수
def predict_and_display(image):
    st.image(image, caption="분석할 식물 사진", use_container_width=True)
    
    if model is None:
        st.error("🚨 plant_model.pkl 파일이 없어서 AI 진단을 할 수 없습니다!")
        return

    with st.spinner("🔍 AI가 식물 상태를 분석 중입니다..."):
        features = extract_features(image)
        prediction = model.predict([features])[0]
        
    st.divider()
    st.markdown("### 🤖 AI 진단 결과")
    
    if prediction == 'healthy':
        st.success("✅ **상태: 건강함 (Healthy)**")
        st.info("💡 식물이 아주 튼튼하게 잘 자라고 있습니다! 현재 관리 루틴을 유지해 주세요.")
    elif prediction == 'disease':
        st.error("⚠️ **상태: 병충해 의심 (Disease)**")
        st.warning("💡 잎에 반점이 감지되었습니다. 잎 뒤쪽을 확인하고 살균제를 도포해 주세요.")
    elif prediction == 'sunlight':
        st.warning("☀️ **상태: 일조량 부족 (Sunlight Needed)**")
        st.info("💡 잎이 누렇게 변하고 있습니다. 조금 더 밝은 창가 쪽으로 이동시켜 주세요.")
    elif prediction == 'water':
        st.warning("💧 **상태: 수분 부족 (Water Needed)**")
        st.info("💡 잎 테두리가 마르고 있습니다. 흙 상태를 확인하고 즉시 물을 듬뿍 주세요.")

# 3. 화면 구성
st.metric(label="토양 습도", value="45%", delta="정상")
st.divider()

st.markdown("### 📸 식물 상태 점검하기")
tab1, tab2 = st.tabs(["📷 실시간 촬영", "📁 파일 업로드"])

with tab1:
    camera_image = st.camera_input("식물 사진을 찍어주세요")
    if camera_image:
        image = Image.open(camera_image)
        predict_and_display(image)

with tab2:
    uploaded_file = st.file_uploader("식물 이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        predict_and_display(image)
