import streamlit as st
import pandas as pd

from src.data_ops import load_csv, generate_sample_data
from src.profile_ops import generate_profile_html


st.set_page_config(page_title="Profiling", page_icon="📑", layout="wide")


def sidebar_data_source():
    st.sidebar.header("데이터 소스")
    source = st.sidebar.radio("선택", ["CSV 업로드", "샘플 데이터 사용"], index=1)

    if source == "CSV 업로드":
        uploaded = st.sidebar.file_uploader("CSV 파일 업로드", type=["csv"]) 
        sep = st.sidebar.text_input("구분자(sep)", value=",", help="일반적으로 , 또는 ;")
        decimal = st.sidebar.text_input("소수점 문자(decimal)", value=".")
        encoding = st.sidebar.text_input("인코딩", value="utf-8")
        if uploaded is not None:
            df = load_csv(uploaded, sep=sep or ",", decimal=decimal or ".", encoding=encoding or "utf-8")
        else:
            df = None
    else:
        rows = st.sidebar.slider("샘플 데이터 행 수", min_value=100, max_value=10000, value=1500, step=100)
        seed = st.sidebar.number_input("시드", value=42, step=1)
        df = generate_sample_data(rows=rows, seed=int(seed))
    return df


def main():
    st.title("📑 데이터 프로파일링")
    st.caption("ydata-profiling을 사용하여 자동 리포트를 생성합니다.")

    df = sidebar_data_source()
    if df is None or df.empty:
        st.info("좌측 사이드바에서 데이터를 선택하세요.")
        st.stop()

    st.markdown("### 옵션")
    c1, c2 = st.columns(2)
    with c1:
        minimal = st.checkbox("Minimal 모드", value=True, help="빠른 생성, 항목 일부 생략")
    with c2:
        max_sample = min(len(df), 20000)
        sample_n = st.slider("샘플 크기(0=전체)", 0, max_sample, min(5000, max_sample))

    generate = st.button("프로파일 생성")
    if not generate:
        st.info("옵션을 설정하고 '프로파일 생성'을 클릭하세요.")
        st.stop()

    with st.spinner("리포트 생성 중..."):
        try:
            html = generate_profile_html(df, minimal=minimal, sample_n=sample_n or None)
        except ImportError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"생성 실패: {e}")
            st.stop()

    st.markdown("### 리포트")
    st.components.v1.html(html, height=900, scrolling=True)

    st.download_button(
        label="HTML 다운로드",
        data=html.encode("utf-8"),
        file_name="profiling_report.html",
        mime="text/html",
    )


if __name__ == "__main__":
    main()

