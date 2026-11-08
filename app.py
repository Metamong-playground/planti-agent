def generate_planti_report(spot_count, yellow_ratio, marginal_necrosis_area, total_leaf_area):
    """
    학술 기준(Bousset 2016, Zhu 2024, Nevin 1987) 기반 픽셀 텐서 분석 후
    직관적인 피드백 메시지를 반환하는 진단 모듈
    """
    
    # 1. 병충해 (Disease): 소형 반점(6~320px) 개수 기준
    # (Bousset et al., 2016 - Lesions per leaf)
    is_disease = spot_count >= 5  # 프로젝트 기준 임계값
    
    # 2. 일조량 부족 (Sunlight Deficiency): HSV 황화 비율 기준
    # (Zhu et al., 2024 - Chlorosis Yellowing Index)
    is_sunlight_deficient = yellow_ratio >= 0.25  # 전체 잎 대비 황화 영역 25% 이상
    
    # 3. 수분 부족 (Water Stress): 잎 가장자리 대형 갈변 패치(320px 초과) 기준
    # (Nevin & Lovatt, 1987 - Marginal Necrosis)
    is_water_deficient = marginal_necrosis_area >= 320 or (marginal_necrosis_area / total_leaf_area) >= 0.15
    
    # --- 상태별 출력 메시지 할당 ---
    messages = []
    
    if is_disease:
        messages.append("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
        
    if is_sunlight_deficient:
        messages.append("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
        
    if is_water_deficient:
        messages.append("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
        
    # 이상 증상이 하나도 감지되지 않은 경우 (Healthy)
    if not messages:
        messages.append("정상: 식물이 건강한 상태입니다. 현재의 환경과 관리를 지속해 주세요.")
        
    # 결과 출력
    return "\n".join(messages)


# --- 실행 테스트 예시 ---
if __name__ == "__main__":
    # 예
