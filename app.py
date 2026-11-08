def generate_planti_report(spot_count, yellow_ratio, marginal_necrosis_area, total_leaf_area):
    """
    학술 기준 기반 픽셀 텐서 분석 후 피드백 메시지를 반환하는 모듈
    """
    is_disease = spot_count >= 5
    is_sunlight_deficient = yellow_ratio >= 0.25
    is_water_deficient = marginal_necrosis_area >= 320 or (total_leaf_area > 0 and (marginal_necrosis_area / total_leaf_area) >= 0.15)
    
    messages = []
    
    # 40번 줄 근처 if문 블록들
    if is_disease:
        messages.append("병충해: 병충해가 있어요. 바로 다른 화분과 격리시키고, 매일 매일 꼼꼼히 기생충이나 기생충의 알 등을 없애 주세요!")
        
    if is_sunlight_deficient:
        messages.append("일사량 부족: 햇빛의 양이 부족해졌습니다. 식물에게 태양 빛을 공급해주세요")
        
    if is_water_deficient:
        messages.append("물 부족: 물이 부족해졌습니다. 식물에게 물을 더 주세요")
        
    if not messages:
        messages.append("정상: 식물이 건강한 상태입니다. 현재의 환경과 관리를 지속해 주세요.")
        
    return "\n".join(messages)
