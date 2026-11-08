import streamlit as st
import cv2
import numpy as np

# --- 1. Planti 룰-베이스 진단 메시지 생성 모듈 ---
def generate_planti_report(spot_count, yellow_ratio, marginal_necrosis_area, total_leaf_area):
    """
    학술 기준(Bousset 2016, Zhu 2024, Nevin 1987) 기반 픽셀 텐서 분석 후
    정갈하고 깔끔한 진단 메시지를 반환하는 모듈
    """
    # 1. 병충해 (Disease): 소형 반점(6~320px) 5개 이상 (Bousset et al., 2016)
    is_disease = spot_count >= 5
    
    # 2. 일조량 부족 (Sunlight Deficiency): HSV 황화 비율 25% 이상 (Zhu et al., 2024)
    is_sunlight_deficient = yellow_ratio >= 0.25
    
    # 3. 수분 부족 (Water Stress): 잎 가장자리 대형 갈변 패치(320px 초과) (Nevin & Lovatt, 1987)
    is_water_deficient = marginal_necrosis_area >= 320 or (total_leaf_area > 0 and (marginal_necrosis_area / total_leaf_area) >= 0.15)
    
    # 진단 메시지 구성
    messages = []
    
    if is_disease:
        messages.append("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
        
    if is_sunlight_deficient:
        messages.append("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
        
    if is_water_deficient:
        messages.append("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
        
    # 세 가지 이상 증상이 모두 없는 경우 (Healthy)
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
    
    # 잎 전체 영역 마스크 (전체 잎 면적 계산용 - GLAI 기준)
    lower_leaf = np.array([15, 30, 30])
    upper_leaf = np.array([90, 255, 255])
    leaf_mask = cv2.inRange(hsv, lower_leaf, upper_leaf)
    total_leaf_area = cv2.countNonZero(leaf_mask)
    
    # 0 나누기 방지
    if total_leaf_area == 0:
        total_leaf_area = 1
        
    # A. 황화 현상 추출 (Zhu et al., 2024 - Chlorosis Yellowing Index)
    lower_yellow = np.array([15, 80, 80])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_pixels = cv2.countNonZero(yellow_mask)
    yellow_ratio = yellow_pixels / total_leaf_area
    
    # B. 어두운 갈색/괴사 영역 추출 (Bousset 2016, Nevin 1987)
    lower_brown = np.array([0, 40, 20])
    upper_brown = np.array([20, 255, 150])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # 윤곽선(Contours) 검출을 통한 픽셀 크기별 분리
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    spot_count = 0
    marginal_necrosis_area = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 소형 병반 (6px ~ 320px) -> 병충해 반점 카운트 (Bousset 2016)
        if 6 <= area <= 320:
            spot_count += 1
        # 대형 패치 (> 320px) -> 수분 부족 테두리 괴사 면적 합산 (Nevin 1987)
        elif area > 320:
            marginal_necrosis_area += area
            
    # 최종 진단 메시지 생성
    report_text = generate_planti_report(spot_count, yellow_ratio, marginal_necrosis_area, total_leaf_area)
    
    return img, report_text


# --- 3. Streamlit 메인 UI ---
st.title("Planti: 식물 상태 진단 시스템")
st.write("학술 논문 기반 룰-베이스 식물 잎 상태 분석 파이프라인")

uploaded_file = st.file_uploader("식물 잎 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="업로드된 식물 이미지", use_column_width=True)
    
    with st.spinner("픽셀 텐서 분석 중
