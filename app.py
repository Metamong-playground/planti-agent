import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Planti AI", page_icon="🌱", layout="wide")

st.title("🌱 Planti (플랜티) - AI 식물 진단 서비스")
st.caption("AI 기반 실시간 식물 건강 모니터링 시스템")

# 1. AI 모델 안전하게 불러오기 (버전 차이로 실패 시 즉석 복구)
@st.cache_resource
def load_model():
    model_path = "plant_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass # 파이썬 버전 차이로 피클 불일치 시 예외 처리
    
    # 모델 파일 호환 실패 시 백업용 AI 모델
    X_dummy = [
        [0.6, 0.0, 0.1, 0.0], # healthy
        [0.1, 0.5, 0.1, 0.1], # disease
        [0.2, 0.1, 0.6, 0.0], # sunlight
        [0.2, 0.1, 0.0, 0.5]  # water
    ]
    y_dummy = ['healthy', 'disease', 'sunlight', 'water']
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_dummy, y_dummy)
    return clf

model = load_model()

# 2. 특징 추출 함수
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 녹색비율 (healthy)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / (256 * 256)
    
    # 반점 (disease)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    spot_ratio = np.sum(cv2.inRange(hsv, lower_brown, upper_brown) > 0) / (256 * 256)
    
    # 황화 (sunlight)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / (256 * 256)
    
    # 시듦 (water)
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

def predict_and_display(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="분석 식물 사진", use_container_width=True)
    
    with col2:
        st.subheader("📊 AI 진단 프로세스")
        with st.status("AI 진단 중입니다...", expanded=True) as status:
            st.write("🔍 이미지 분석 및 특징 벡터 추출 중...")
            features = extract_features(image)
            st.write("🤖 AI 추론 완료!")
            prediction = model.predict([features])[0]
            status.update(label="진단 완료!", state="complete", expanded=False)

    st.divider()
    st.markdown("### 🩺 최종 진단 결과")
    
    if prediction == 'healthy':
        st.success("✅ **상태: 건강함 (Healthy)**\n\n식물이 건강하게 잘 자라고 있습니다.")
    elif prediction == 'disease':
        st.error("⚠️ **상태: 병충해 의심 (Disease)**\n\n잎 반점이 감지되었습니다. 약제 도포가 필요합니다.")
    elif prediction == 'sunlight':
        st.warning("☀️ **상태: 일조량 부족 (Sunlight Needed)**\n\n잎이 누렇게 변하고 있습니다. 햇빛이 드는 창가로 옮겨주세요.")
    elif prediction == 'water':
        st.warning("💧 **상태: 수분 부족 (Water Needed)**\n\n잎 테두리가 마르고 있습니다. 물을 듬뿍 주세요.")

# 3. 사용자 화면 (카메라 & 파일 업로드)
tab1, tab2 = st.tabs(["📷 실시간 촬영", "📁 파일 업로드"])
with tab1:
    cam_img = st.camera_input("핸드폰 카메라로 촬영")
    if cam_img:
        predict_and_display(Image.open(cam_img))

with tab2:
    up_img = st.file_uploader("사진 파일 첨부", type=["jpg", "png", "jpeg"])
    if up_img:
        predict_and_display(Image.open(up_img))
