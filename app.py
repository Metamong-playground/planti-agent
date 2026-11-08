import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os
import json
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

# --- 0. 로컬 파일 기반 영구 저장 기능 (새로고침/재방문해도 유지) ---
HISTORY_FILE = "planti_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history_data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        pass

if "history" not in st.session_state:
    st.session_state.history = load_history()

# --- 1. AI 모델 로드 (예외 처리 및 백업) ---
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

# --- 2. 이미지 면적 비례 동적 반점 및 생리학 지표 추출 함수 ---
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

# --- 3. 진단 및 모니터링 출력 모듈 ---
def predict_and_display(image):
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="분석 식물 사진", use_container_width=True)
    
    with col2:
        st.subheader("진단 결과")
        with st.status("해상도 비율 기반 픽셀 텐서 분석 중...", expanded=True) as status:
            features = extract_features(image)
            spot_cnt = int(features[1])
            water_score = features[3]
            large_patch = features[4]
            
            st.write(f"검출된 병반 점(6~320px): `{spot_cnt}개`")
            st.write(f"대형 시듦/바램 면적 비율: `{large_patch*100:.2f}%`")
            st.write(f"테두리 수분 스트레스 지수: `{water_score:.3f}`")
            
            # 학술 임계값 기반 계층 판단 로직
            if spot_cnt >= 5 and large_patch < 0.03:
                prediction = 'disease'
                result_label = '병충해'
            elif water_score >= 0.10 or large_patch >= 0.03:
                prediction = 'water'
                result_label = '물 부족'
            else:
                prediction = model.predict([features[:4]])[0]
                label_map = {'healthy': '정상', 'sunlight': '일사량 부족', 'disease': '병충해', 'water': '물 부족'}
                result_label = label_map.get(prediction, '정상')
                
            status.update(label="진단 완료", state="complete", expanded=False)

    st.divider()
    st.markdown("### 최종 진단 결과")
    
    # 이모지 제거 및 지정된 진단 텍스트 반영
    if prediction == 'disease':
        st.error("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
    elif prediction == 'water':
        st.warning("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
    elif prediction == 'sunlight':
        st.warning("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
    elif prediction == 'healthy':
        st.success("정상: 식물이 건강한 상태입니다. 현재의 환경과 관리를 지속해 주세요.")

    # 세션 및 파일(JSON)에 진단 이력 영구 저장
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = {
        "time": now_str,
        "result": result_label,
        "spots": spot_cnt,
        "water": f"{water_score:.2f}"
    }
    
    # 중복 저장 방지 체크
    if not st.session_state.history or st.session_state.history[0]["time"] != now_str:
        st.session_state.history.insert(0, new_entry)
        save_history(st.session_state.history)

# --- 4. 사이드바 - 최근 진단 이력 ---
with st.sidebar:
    st.header("최근 진단 이력")
    if not st.session_state.history:
        st.write("진단 기록 없음")
    else:
        for idx, item in enumerate(st.session_state.history):
            st.markdown(f"**[{item['time']}] {item['result']}**")
            st.caption(f"반점: {item['spots']}개 | 수분지수: {item['water']}")
            st.divider()
        if st.button("이력 전체 삭제"):
            st.session_state.history = []
            save_history([])
            st.rerun()

# --- 5. Streamlit 메인 UI 탭 ([진단 확인] 버튼 적용) ---
tab1, tab2 = st.tabs(["실시간 촬영", "파일 업로드"])

with tab1:
    cam_img = st.camera_input("카메라로 촬영")
    if cam_img:
        if st.button("촬영한 사진 진단 확인", key="btn_cam"):
            predict_and_display(Image.open(cam_img))

with tab2:
    up_img = st.file_uploader("사진 파일 첨부", type=["jpg", "png", "jpeg"])
    if up_img:
        if st.button("첨부한 사진 진단 확인", key="btn_file"):
            predict_and_display(Image.open(up_img))
