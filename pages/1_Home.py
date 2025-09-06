import streamlit as st

from src.settings import APP_NAME, TAGLINE
from src.ui import render_header, render_topnav, render_footer
from src.styles import apply_base_theme


st.set_page_config(page_title=f"{APP_NAME} | Home", page_icon="🏠", layout="wide")


def hero():
    apply_base_theme()
    col1, col2 = st.columns([7, 5])
    with col1:
        st.markdown(
            f"""
            <div style='padding:0.2rem 0 0.2rem 0;'>
              <div style='font-size:2rem;font-weight:800;line-height:1.2;color:#111827;'>
                {APP_NAME}
              </div>
              <div style='font-size:1.05rem;color:#6B7280;margin-top:0.4rem;'>
                {TAGLINE or '데이터를 빠르게 탐색·시각화·모델링하세요.'}
              </div>
              <div style='margin-top:0.9rem;display:flex;gap:0.6rem;'>
                
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        cta1, cta2 = st.columns([1, 1])
        with cta1:
            st.page_link("app.py", label="📊 Open Studio")
        with cta2:
            st.page_link("pages/5_Profiling.py", label="📑 Open Profiling")
    with col2:
        st.info("샘플 CSV를 업로드하거나, Studio에서 샘플 데이터를 생성해 바로 시작해보세요.")


def features():
    st.markdown("### 주요 기능")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("EDA")
        st.write("미리보기/요약/결측/상관 분석을 간편하게.")
    with c2:
        st.subheader("시각화")
        st.write("히스토그램/산점/박스/막대/시계열 및 고급 플롯.")
    with c3:
        st.subheader("모델링")
        st.write("Baseline+튜닝(Grid/Random+CV), 성능 리포트.")


def main():
    render_header(icon="🏠", show_tagline=True)
    render_topnav(active="Home")
    hero()
    st.divider()
    features()
    render_footer()


if __name__ == "__main__":
    main()

