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

# 1. AI 모델 불러오기 (예외 처리 포함)
@st.cache_resource
def load_model():
    model_path = "plant_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    
    # 임시 백업 모델
    X_dummy = [
        [0.6, 1.0, 0.1, 0.0], # healthy (점 개수 적음)
        [0.1, 15.0, 0.1, 0.1], # disease (점 개수 많음)
        [0.2, 2.0, 0.6, 0.0], # sunlight
        [0.2, 2.0, 0.0, 0.5]  # water
    ]
    y_dummy = ['healthy', 'disease', 'sunlight', 'water']
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_dummy, y_dummy)
    return clf

model = load_model()

# 2. 개별 '점(Spot 덩어리)' 검출 기반 특징 추출 함수
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1) 전체 녹색 비율 (healthy)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / (256 * 256)
    
    # 2) "점(Spot)" 개수 검출 (disease)
    # 갈색 ~ 어두운 검은색 영역 추출
    lower_brown = np.array([5, 40, 20])
    upper_brown = np.array([30, 255, 200])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 노이즈 제거 (가우시안 블러)
    blurred = cv2.GaussianBlur(brown_mask, (5, 5), 0)
    
    # 뭉쳐있는 점(Contour) 덩어리 찾기
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 점으로 판단할 최소/최대 면적 기준 (픽셀 단위)
    spot_count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 <= area <= 1500:  # 너무 미세한 잡영 제외, 잎 전체 크기보다 작은 반점만 카운트
            spot_count += 1
            
    # 3) 황화 현상 (sunlight)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / (256 * 256)
    
    # 4) 시듦 현상 (water)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    edge_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    water_need = 0.0
    if edge_contours:
        cnt = max(edge_contours, key=cv2.contourArea)
        mask_edge = np.zeros((256, 256), dtype=np.uint8)
        cv2.drawContours(mask_edge, [cnt], -1, 255, thickness=15)
        edge_brown = cv2.bitwise_and(brown_mask, mask_edge)
        water_need = np.sum(edge_brown > 0) / (np.sum(mask_edge > 0) + 1e-5)

    return [g_ratio, float(spot_count), sunlight_index, water_need]

def predict_and_display(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="분석 식물 사진", use_container_width=True)
    
    with col2:
        st.subheader("📊 AI 진단 프로세스")
        with st.status("AI가 식물 잎의 점과 색상을 분석 중입니다...", expanded=True) as status:
            features = extract_features(image)
            st.write(f"🔍 **검출된 반점 개수:** `{int(features[1])} 개`")
            st.write(f"📈 **녹색 비율:** `{features[0]*100:.1f}%` / **황화 비율:** `{features[2]*100:.1f}%`")
            
            # 반점 개수가 많으면(5개 이상) 우선적으로 병충해 진단 지원
            if features[1] >= 5:
                prediction = 'disease'
            else:
                prediction = model.predict([features])[0]
                
            status.update(label="진단 완료!", state="complete", expanded=False)

    st.divider()
    st.markdown("### 🩺 최종 진단 결과")
    
    if prediction == 'disease':
        st.error(f"⚠️ **상태: 병충해 의심 (Disease)**\n\n잎에서 **{int(features[1])}개**의 병반(반점) 덩어리가 검출되었습니다! 살균제 처방이 필요합니다.")
    elif prediction == 'healthy':
        st.success("✅ **상태: 건강함 (Healthy)**\n\n식물이 건강하게 잘 자라고 있습니다.")
    elif prediction == 'sunlight':
        st.warning("☀️ **상태: 일조량 부족 (Sunlight Needed)**\n\n잎이 누렇게 변하고 있습니다. 햇빛이 드는 창가로 옮겨주세요.")
    elif prediction == 'water':
        st.warning("💧 **상태: 수분 부족 (Water Needed)**\n\n잎 테두리가 마르고 있습니다. 물을 듬뿍 주세요.")

# 3. 사용자 화면
tab1, tab2 = st.tabs(["📷 실시간 촬영", "📁 파일 업로드"])
with tab1:
    cam_img = st.camera_input("핸드폰 카메라로 촬영")
    if cam_img:
        predict_and_display(Image.open(cam_img))

with tab2:
    up_img = st.file_uploader("사진 파일 첨부", type=["jpg", "png", "jpeg"])
    if up_img:
        predict_and_display(Image.open(up_img))
