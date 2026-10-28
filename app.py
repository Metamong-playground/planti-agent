import streamlit as st

st.set_page_config(page_title="Planti", page_icon="🌱")

st.title("🌱 Planti (플랜티)")
st.subheader("AI 기반 식물 진단 & 모니터링")

st.metric(label="토양 습도", value="45%", delta="정상")
st.info("💡 식물 상태가 아주 양호합니다!")
