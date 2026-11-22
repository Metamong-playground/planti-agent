import streamlit as st
from PIL import Image
import joblib
import cv2
import numpy as np
import os
import json
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Planti AI", layout="wide")

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

st.title("🪴 Planti - 식물 상태 진단 에이전트")
st.caption("식물 건강을 간편하게 확인")

SCORE_MAP = {'정상': 100, '일사량 부족': 60, '물 부족': 60, '병충해': 30}

HISTORY_FILE = "planti_history.json"


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if "score" not in item or item["score"] is None:
                    item["score"] = SCORE_MAP.get(item.get("result", "정상"), 100)
            return data
        except Exception:
            return []
    return []


def save_history(history_data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


if "history" not in st.session_state:
    st.session_state.history = load_history()


@st.cache_resource
def load_model():
    model_path = "plant_model.pkl"
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass

    X_dummy = [
        [0.80, 0.0, 0.02, 0.01, 0.00],
        [0.40, 9.0, 0.05, 0.05, 0.08],
        [0.15, 0.0, 0.65, 0.03, 0.00],
        [0.45, 1.0, 0.05, 0.35, 0.01]
    ]
    y_dummy = ['healthy', 'disease', 'sunlight', 'water']
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_dummy, y_dummy)
    return clf


model = load_model()


def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    leaf_mask = ((s > 35) & (v > 20)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    total_leaf_pixels = np.sum(leaf_mask > 0)
    if total_leaf_pixels < 200:
        leaf_mask = np.ones((256, 256), dtype=np.uint8) * 255
        total_leaf_pixels = 256 * 256

    lower_green = np.array([36, 45, 35])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum((cv2.inRange(hsv, lower_green, upper_green) > 0) & (leaf_mask > 0)) / total_leaf_pixels

    lower_yellow = np.array([20, 45, 110])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum((cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) & (leaf_mask > 0)) / total_leaf_pixels

    inner_leaf = cv2.erode(leaf_mask, np.ones((23, 23), np.uint8), iterations=1)
    edge_mask = cv2.subtract(leaf_mask, inner_leaf)
    total_edge_pixels = max(np.sum(edge_mask > 0), 1)

    lower_brown = np.array([5, 45, 35])
    upper_brown = np.array([19, 255, 185])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    water_need = np.sum((brown_mask > 0) & (edge_mask > 0)) / total_edge_pixels

    black_spot_mask = cv2.inRange(hsv, np.array([0, 20, 15]), np.array([25, 255, 85]))
    spot_in_leaf = cv2.bitwise_and(black_spot_mask, inner_leaf)
    blurred = cv2.GaussianBlur(spot_in_leaf, (3, 3), 0)
    contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_spot_area = total_leaf_pixels * 0.0003
    max_spot_area = total_leaf_pixels * 0.03
    spot_count = 0
    spot_area_sum = 0.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_spot_area <= area <= max_spot_area:
            spot_count += 1
            spot_area_sum += area

    large_patch_ratio = spot_area_sum / total_leaf_pixels

    return [g_ratio, float(spot_count), sunlight_index, water_need, large_patch_ratio]


def predict_and_display(image):
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="분석 식물 사진", use_container_width=True)

    with col2:
        st.subheader("진단 결과")
        with st.status("해상도 비율 기반 픽셀 텐서 분석 중...", expanded=True) as status:
            features = extract_features(image)
            g_ratio = features[0]
            spot_cnt = int(features[1])
            sunlight_idx = features[2]
            water_score = features[3]
            large_patch = features[4]

            st.write(f"검출된 병반 점(6~320px): `{spot_cnt}개`")
            st.write(f"대형 시듦/바램 면적 비율: `{large_patch*100:.2f}%`")
            st.write(f"테두리 수분 스트레스 지수: `{water_score:.3f}`")

            if spot_cnt >= 4 or large_patch >= 0.035:
                prediction = 'disease'
            elif sunlight_idx >= 0.25 and sunlight_idx > water_score:
                prediction = 'sunlight'
            elif water_score >= 0.14:
                prediction = 'water'
            elif g_ratio >= 0.65 and spot_cnt <= 2 and water_score < 0.10:
                prediction = 'healthy'
            else:
                try:
                    prediction = model.predict([features])[0]
                except Exception:
                    prediction = 'healthy'

            label_map = {'healthy': '정상', 'sunlight': '일사량 부족', 'disease': '병충해', 'water': '물 부족'}
            result_label = label_map.get(prediction, '정상')
            score = SCORE_MAP.get(result_label, 100)

            status.update(label="진단 완료", state="complete", expanded=False)

        st.divider()
        st.markdown("### 최종 진단 결과")

        if prediction == 'disease':
            st.error("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
        elif prediction == 'water':
            st.warning("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
        elif prediction == 'sunlight':
            st.warning("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
       elif prediction == 'healthy':
            st.success("정상: 식물이 건강한 상태입니다. 현재의 환경과 관리를 지속해 주세요.")
