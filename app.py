import streamlit as st
import cv2
import numpy as np

# --- 1. Planti 룰-베이스 진단 및 메시지 생성 모듈 ---
def generate_planti_report(spot_count, yellow_ratio, marginal_necrosis_area, total_leaf_area):
    """
    학술 기준(Bousset 2016, Zhu 2024, Nevin 1987) 기반 픽셀 텐서 분석 후
    직관적인 피드백 메시지를 반환하는 진단 모듈
    """
    # 학술 기준 기반 임계값(Threshold) 조건 판정
    # 1. 병충해: 소형 반점(6~320px) 5개 이상 (Bousset et al., 2016)
    is_disease = spot_count >= 5
    
    # 2. 일조량 부족: HSV 황화 비율 25% 이상 (Zhu et al., 2024)
    is_sunlight_deficient = yellow_ratio >= 0.25
    
    # 3. 수분 부족: 잎 가장자리 대형 갈변 패치(320px 초과) (Nevin & Lovatt, 1987)
    is_water_deficient = marginal_necrosis_area >= 320 or (total_leaf_area > 0 and (marginal_necrosis_area / total_leaf_area) >= 0.15)
    
    # 진단 메시지 구성
    messages = []
    
    if is_disease:
        messages.append("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
        
    if is_sunlight_deficient:
        messages.append("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
        
    if is_water_deficient:
        messages.append("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
        
    # 세 가지 이상 증상이 모두 없는 경우
    if not messages:
        messages.append("정상: 식물이 건강한 상태입니다. 현재의 환경과 관리를 지속해 주세요.")
        
    return "\n\n".join(messages)


# --- 2. OpenCV 픽셀 특성 추출 파이프라인 ---
def analyze_leaf_image(image_bytes):
    # 이미지 디코딩
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return None, "이미지를 불러올 수 없습니다."

    # RGB -> HSV 변환
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 잎 전체 영역 마스크 (전체 잎 면적 계산용)
    # 일반적인 녹색~연두색~황색 파장 범위
    lower_leaf = np.array([15, 30, 30])
    upper_leaf = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    total_leaf_area = cv2.countNonZero(leaf_mask)
    
    if total_leaf_area == 0:
