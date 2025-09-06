import streamlit as st
import pandas as pd
import numpy as np
from src.data_ops import generate_sample_data, load_csv, detect_column_types
from src.eda_ops import basic_info, missing_summary, numeric_summary, correlation_matrix
from src.viz_ops import (
    plot_histogram,
    plot_scatter,
    plot_box,
    plot_bar_count,
    plot_line,
    plot_correlation_heatmap,
)


st.set_page_config(
    page_title="Data Analysis Starter",
    page_icon="📊",
    layout="wide",
)


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
        rows = st.sidebar.slider("샘플 데이터 행 수", min_value=100, max_value=5000, value=500, step=100)
        seed = st.sidebar.number_input("시드", value=42, step=1)
        df = generate_sample_data(rows=rows, seed=int(seed))

    return df


def show_overview(df: pd.DataFrame):
    st.subheader("개요")
    info = basic_info(df)

    col1, col2, col3 = st.columns(3)
    col1.metric("행 수", f"{info['rows']:,}")
    col2.metric("열 수", f"{info['columns']:,}")
    col3.metric("메모리 사용량", info['memory'])

    with st.expander("데이터 미리보기", expanded=True):
        st.dataframe(df.head(30), use_container_width=True)

    with st.expander("결측치 현황"):
        st.dataframe(missing_summary(df), use_container_width=True)

    num_cols, cat_cols, dt_cols = detect_column_types(df)
    if len(num_cols) >= 2:
        st.markdown("#### 상관관계 히트맵")
        corr = correlation_matrix(df[num_cols])
        fig = plot_correlation_heatmap(corr)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("수치형 컬럼이 2개 이상일 때 상관관계를 표시합니다.")


def show_explore(df: pd.DataFrame):
    st.subheader("탐색")
    num_cols, cat_cols, dt_cols = detect_column_types(df)

    tabs = st.tabs(["수치형 요약", "범주형 분포"]) 

    with tabs[0]:
        if num_cols:
            st.dataframe(numeric_summary(df[num_cols]), use_container_width=True)
        else:
            st.info("수치형 컬럼이 없습니다.")

    with tabs[1]:
        if cat_cols:
            col = st.selectbox("컬럼 선택", cat_cols)
            vc = df[col].value_counts(dropna=False).rename_axis(col).to_frame("count")
            st.dataframe(vc, use_container_width=True)
        else:
            st.info("범주형 컬럼이 없습니다.")


def show_visualize(df: pd.DataFrame):
    st.subheader("시각화")
    num_cols, cat_cols, dt_cols = detect_column_types(df)

    chart_type = st.selectbox(
        "차트 종류",
        ["히스토그램", "산점도", "박스플롯", "막대(빈도)", "선형(시계열)"]
    )

    if chart_type == "히스토그램":
        if not num_cols:
            st.warning("수치형 컬럼이 필요합니다.")
            return
        x = st.selectbox("X(수치형)", num_cols)
        color = st.selectbox("색상(옵션)", [None] + cat_cols)
        bins = st.slider("빈 개수", 5, 100, 30)
        fig = plot_histogram(df, x=x, color=color, nbins=bins)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "산점도":
        if len(num_cols) < 2:
            st.warning("수치형 컬럼이 2개 이상 필요합니다.")
            return
        x = st.selectbox("X", num_cols, index=0)
        y = st.selectbox("Y", [c for c in num_cols if c != x], index=0)
        color = st.selectbox("색상(옵션)", [None] + cat_cols)
        fig = plot_scatter(df, x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "박스플롯":
        if not num_cols:
            st.warning("수치형 컬럼이 필요합니다.")
            return
        y = st.selectbox("Y(수치형)", num_cols)
        x = st.selectbox("X(범주형)", cat_cols) if cat_cols else None
        color = st.selectbox("색상(옵션)", [None] + (cat_cols if cat_cols else []))
        fig = plot_box(df, x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "막대(빈도)":
        if not cat_cols:
            st.warning("범주형 컬럼이 필요합니다.")
            return
        x = st.selectbox("X(범주)", cat_cols)
        color = st.selectbox("색상(옵션)", [None] + [c for c in cat_cols if c != x])
        fig = plot_bar_count(df, x=x, color=color)
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "선형(시계열)":
        candidate_x = dt_cols if dt_cols else num_cols
        if not candidate_x or not num_cols:
            st.warning("X(시간/수치) 1개와 Y(수치) 1개가 필요합니다.")
            return
        x = st.selectbox("X(시간/수치)", candidate_x)
        y = st.selectbox("Y(수치)", num_cols)
        color = st.selectbox("색상(옵션)", [None] + cat_cols)
        fig = plot_line(df.sort_values(by=x), x=x, y=y, color=color)
        st.plotly_chart(fig, use_container_width=True)


def show_transform(df: pd.DataFrame):
    st.subheader("변환/내보내기")
    cols = st.multiselect("유지할 컬럼 선택", df.columns.tolist(), default=df.columns.tolist())
    out = df[cols].copy() if cols else df.copy()

    dropna = st.checkbox("결측치 행 제거", value=False)
    if dropna:
        out = out.dropna()

    query_str = st.text_input("행 필터 쿼리(pandas .query 문법)", value="")
    if query_str.strip():
        try:
            out = out.query(query_str)
        except Exception as e:
            st.error(f"쿼리 오류: {e}")

    sample_n = st.slider("샘플 수(0=전체)", min_value=0, max_value=min(5000, len(out)), value=0)
    if sample_n and sample_n < len(out):
        out = out.sample(n=sample_n, random_state=42)

    st.markdown("#### 결과 미리보기")
    st.dataframe(out.head(100), use_container_width=True)

    csv = out.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="CSV 다운로드",
        data=csv,
        file_name="transformed.csv",
        mime="text/csv",
    )


def show_model_stub(df: pd.DataFrame):
    st.subheader("모델(Stub)")
    st.info("간단한 베이스라인 모델을 이후에 추가할 영역입니다. 예: 회귀/분류")
    with st.expander("힌트: 다음 기능을 고려해보세요"):
        st.markdown(
            "- 타깃 컬럼 선택 및 train/test 분할\n"
            "- 스케일링 및 인코딩\n"
            "- 간단한 알고리즘(선형회귀/로지스틱/랜덤포레스트) 학습\n"
            "- 기본 평가지표 출력 (RMSE/Accuracy 등)"
        )


def main():
    st.title("📊 데이터 분석 스타터 (Streamlit)")
    st.caption("CSV 업로드 또는 샘플 데이터로 간단한 EDA와 시각화를 수행합니다.")

    df = sidebar_data_source()
    if df is None or df.empty:
        st.warning("좌측 사이드바에서 데이터를 불러오세요.")
        st.stop()

    tabs = st.tabs(["개요", "탐색", "시각화", "변환", "모델(Stub)"])
    with tabs[0]:
        show_overview(df)
    with tabs[1]:
        show_explore(df)
    with tabs[2]:
        show_visualize(df)
    with tabs[3]:
        show_transform(df)
    with tabs[4]:
        show_model_stub(df)


if __name__ == "__main__":
    main()

