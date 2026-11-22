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

st.title("🪴 Planti - 식물 상태 진단 에이전트")
st.caption("식물 건강을 간편하게 확인")

SCORE_MAP = {'정상': 100, '일사량 부족': 60, '물 부족': 60, '병충해': 30}

# --- 0. 로컬 파일 기반 영구 저장 기능 (이전 데이터 보완 처리) ---
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

# --- 1. AI 모델 로드 ---
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

# --- 2. 특성 추출 함수 (4가지 핵심 포인트 반영) ---
def extract_features(img_pil):
    img = np.array(img_pil.convert('RGB'))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 배경 제거 후 순수 잎 영역만 추출
    leaf_mask = ((s > 35) & (v > 20)).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_OPEN, kernel)

    total_leaf_pixels = np.sum(leaf_mask > 0)
    if total_leaf_pixels < 200:
        leaf_mask = np.ones((256, 256), dtype=np.uint8) * 255
        total_leaf_pixels = 256 * 256

    # 1) 완벽 잎: 깨끗 초록 비율
    lower_green = np.array([36, 45, 35])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum((cv2.inRange(hsv, lower_green, upper_green) > 0) & (leaf_mask > 0)) / total_leaf_pixels

    # 2) 햇빛 부족: 연노랑 비율
    lower_yellow = np.array([20, 45, 110])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum((cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) & (leaf_mask > 0)) / total_leaf_pixels

    # 3) 물 부족: 잎 끝(테두리) 갈색 비율
    inner_leaf = cv2.erode(leaf_mask, np.ones((23, 23), np.uint8), iterations=1)
    edge_mask = cv2.subtract(leaf_mask, inner_leaf)
    total_edge_pixels = max(np.sum(edge_mask > 0), 1)

    lower_brown = np.array([5, 45, 35])
    upper_brown = np.array([19, 255, 185])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    water_need = np.sum((brown_mask > 0) & (edge_mask > 0)) / total_edge_pixels

    # 4) 병해충: 검은 반점 개수 및 면적 비율
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
        if min
