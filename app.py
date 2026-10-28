import streamlit as st
from PIL import Image

st.set_page_config(page_title="Planti", page_icon="🌱")

st.title("🌱 Planti (플랜티)")
st.subheader("AI 기반 식물 진단 & 모니터링")

# 기본 상태 표시
st.metric(label="토양 습도", value="45%", delta="정상")
st.info("💡 식물 상태가 아주 양호합니다!")

st.divider()

# 식물 사진 촬영 및 업로드 세션
st.markdown("### 📸 식물 상태 점검하기")
tab1, tab2 = st.tabs(["📷 실시간 촬영", "📁 파일 업로드"])

with tab1:
    camera_image = st.camera_input("식물 사진을 찍어주세요")
    if camera_image:
        st.image(camera_image, caption="촬영된 식물 사진", use_container_width=True)
        st.success("사진이 성공적으로 캡처되었습니다! AI 분석 준비 완료 ✨")

with tab2:
    uploaded_file = st.file_uploader("식물 이미지 파일을 선택하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 식물 사진", use_container_width=True)
        st.success("이미지가 정상적으로 업로드되었습니다! AI 분석 준비 완료 ✨")
