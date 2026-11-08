with tab3:
    st.subheader("식물 종합 건강 리포트")
    st.write("누적된 진단 히스토리를 분석하여 전반적인 건강지수와 추이 그래프를 생성해.")
    
    if st.button("건강 리포트 및 그래프 생성", key="btn_graph"):
        if not st.session_state.history:
            st.warning("아직 진단 기록이 없어! 식물 사진을 먼저 진단해 줘.")
        else:
            # 히스토리 데이터를 역순(시간순: 과거 -> 최근)으로 정렬
            history_list = st.session_state.history[::-1]
            df = pd.DataFrame(history_list)
            
            # 1) 최근 건강 지수 Metric 출력
            latest_score = df["score"].iloc[-1]
            avg_score = df["score"].mean()
            
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.metric(label="최근 진단 건강지수", value=f"{latest_score}점")
            with col_m2:
                st.metric(label="평균 건강지수", value=f"{avg_score:.1f}점")
            with col_m3:
                st.metric(label="총 진단 횟수", value=f"{len(df)}회")
            
            st.divider()
            
            # 2) 시간별 식물 건강지수 변화 그래프 (첫 번째 항목 누락 문제 해결)
            st.markdown("### 시간별 식물 건강지수 변화 추이 (100점 만점)")
            
            # X축 인덱스를 회차 및 일시로 깔끔하게 정돈
            chart_df = pd.DataFrame({
                "건강지수": df["score"].values
            }, index=[f"{i+1}회차 ({t[11:16]})" for i, t in enumerate(df["time"])])
            
            st.line_chart(chart_df)
            
            st.success("건강 리포트 작성이 완료되었습니다!")
