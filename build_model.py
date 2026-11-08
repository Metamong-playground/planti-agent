import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. OpenCV 특징 추출 함수 (g_ratio, spot_ratio, sunlight_index, water_need)
def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (256, 256))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 녹색 비율 (healthy)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    g_ratio = np.sum(cv2.inRange(hsv, lower_green, upper_green) > 0) / (256 * 256)
    
    # 반점 비율 (disease)
    lower_brown = np.array([10, 50, 20])
    upper_brown = np.array([30, 255, 200])
    spot_ratio = np.sum(cv2.inRange(hsv, lower_brown, upper_brown) > 0) / (256 * 256)
    
    # 황화 수치 (sunlight)
    lower_yellow = np.array([20, 40, 100])
    upper_yellow = np.array([35, 255, 255])
    sunlight_index = np.sum(cv2.inRange(hsv, lower_yellow, upper_yellow) > 0) / (256 * 256)
    
    # 테두리 시듦 수치 (water)
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

# 2. dataset 폴더 순회 및 AI 학습
categories = ['healthy', 'disease', 'sunlight', 'water']
X_data, y_labels = [], []

for category in categories:
    folder_path = os.path.join('dataset', category)
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                features = extract_features(os.path.join(folder_path, file_name))
                if features:
                    X_data.append(features)
                    y_labels.append(category)

if X_data:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_data, y_labels)
    joblib.dump(model, 'plant_model.pkl')
    print("🎉 plant_model.pkl 학습 및 생성 완료!")
