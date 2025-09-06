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
    plot_feature_importance,
    plot_scatter_matrix,
    plot_density_contour,
    plot_density_heatmap,
    plot_histogram_facet,
    plot_violin_facet,
)
from src.model_ops import (
    infer_problem_type,
    train_and_evaluate,
    serialize_model_to_bytes,
    tune_and_evaluate,
    extract_feature_importances,
    align_columns_for_inference,
)
from src.preprocess_ops import iqr_filter, remove_duplicates
from src.settings import APP_NAME, TAGLINE
from src.ui import render_header


st.set_page_config(page_title=APP_NAME, page_icon="📊", layout="wide")
render_header(icon="📊", show_tagline=True)


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

    # Advanced visualization
    with st.expander("고급 시각화", expanded=False):
        adv_type = st.selectbox(
            "종류",
            ["산점도 행렬", "2D 밀도(컨투어)", "2D 밀도(히트맵)", "페이싯 히스토그램", "페이싯 바이올린"],
            index=0,
        )

        if adv_type == "산점도 행렬":
            if len(num_cols) < 2:
                st.info("수치형 컬럼이 2개 이상 필요합니다.")
            else:
                dims = st.multiselect("수치형 컬럼 선택", num_cols, default=num_cols[: min(4, len(num_cols))])
                color = st.selectbox("색상(옵션)", [None] + cat_cols)
                sample_n = st.slider("샘플 수(0=전체)", 0, min(len(df), 5000), min(len(df), 1500))
                dfx = df.sample(n=sample_n, random_state=42) if sample_n and sample_n < len(df) else df
                if len(dims) >= 2:
                    fig = plot_scatter_matrix(dfx, dimensions=dims, color=color)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("2개 이상의 수치형 컬럼을 선택하세요.")

        elif adv_type == "2D 밀도(컨투어)":
            if len(num_cols) < 2:
                st.info("수치형 컬럼이 2개 이상 필요합니다.")
            else:
                x = st.selectbox("X", num_cols)
                y = st.selectbox("Y", [c for c in num_cols if c != x])
                color = st.selectbox("색상(옵션)", [None] + cat_cols)
                fig = plot_density_contour(df, x=x, y=y, color=color)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == "2D 밀도(히트맵)":
            if len(num_cols) < 2:
                st.info("수치형 컬럼이 2개 이상 필요합니다.")
            else:
                x = st.selectbox("X", num_cols)
                y = st.selectbox("Y", [c for c in num_cols if c != x])
                nbx = st.slider("X 빈 개수", 5, 100, 30)
                nby = st.slider("Y 빈 개수", 5, 100, 30)
                fig = plot_density_heatmap(df, x=x, y=y, nbinsx=nbx, nbinsy=nby)
                st.plotly_chart(fig, use_container_width=True)

        elif adv_type == "페이싯 히스토그램":
            x_all = df.columns.tolist()
            x = st.selectbox("X", x_all)
            facet_row = st.selectbox("Facet 행", [None] + cat_cols)
            facet_col = st.selectbox("Facet 열", [None] + cat_cols)
            color = st.selectbox("색상(옵션)", [None] + cat_cols)
            bins = st.slider("빈 개수", 5, 100, 30)
            fig = plot_histogram_facet(df, x=x, facet_row=facet_row, facet_col=facet_col, color=color, nbins=bins)
            st.plotly_chart(fig, use_container_width=True)

        elif adv_type == "페이싯 바이올린":
            if not num_cols:
                st.info("수치형 컬럼이 필요합니다.")
            else:
                y = st.selectbox("Y(수치형)", num_cols)
                x = st.selectbox("X(범주, 옵션)", [None] + cat_cols)
                facet_row = st.selectbox("Facet 행", [None] + cat_cols)
                facet_col = st.selectbox("Facet 열", [None] + cat_cols)
                color = st.selectbox("색상(옵션)", [None] + cat_cols)
                fig = plot_violin_facet(df, y=y, x=x if x else None, facet_row=facet_row, facet_col=facet_col, color=color)
                st.plotly_chart(fig, use_container_width=True)


def show_transform(df: pd.DataFrame):
    st.subheader("변환/내보내기")
    cols = st.multiselect("유지할 컬럼 선택", df.columns.tolist(), default=df.columns.tolist())
    out = df[cols].copy() if cols else df.copy()

    dropna = st.checkbox("결측치 행 제거", value=False)
    if dropna:
        out = out.dropna()

    with st.expander("고급 전처리", expanded=False):
        rm_dups = st.checkbox("중복 제거", value=False)
        if rm_dups:
            dup_subset = st.multiselect("중복 판단 컬럼(없으면 전체)", out.columns.tolist())
            dup_keep_choice = st.selectbox("중복 유지", ["first", "last", "drop all"], index=0)
            keep_param = dup_keep_choice if dup_keep_choice != "drop all" else False
            out = remove_duplicates(out, subset=dup_subset if dup_subset else None, keep=keep_param)

        num_cols, _, _ = detect_column_types(out)
        use_iqr = st.checkbox("IQR 이상치 제거", value=False)
        if use_iqr and num_cols:
            sel = st.multiselect("대상 수치 컬럼", num_cols, default=num_cols)
            kk = st.slider("IQR 배수(k)", 0.5, 3.0, 1.5, 0.1)
            out, stats = iqr_filter(out, sel, k=float(kk))
            with st.expander("IQR 통계", expanded=False):
                st.dataframe(stats, use_container_width=True)

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


def show_model(df: pd.DataFrame):
    st.subheader("모델")
    all_cols = df.columns.tolist()
    if not all_cols:
        st.info("데이터가 비어있습니다.")
        return

    with st.form("model_form"):
        target = st.selectbox("타깃 컬럼", all_cols, index=min(len(all_cols)-1, all_cols.index(all_cols[-1])))
        y = df[target]

        auto_problem = infer_problem_type(y)
        problem_map = {"자동": auto_problem, "분류": "classification", "회귀": "regression"}
        problem_choice = st.selectbox("문제 유형", list(problem_map.keys()), index=0, help=f"자동 추정: {auto_problem}")
        problem = problem_map[problem_choice]

        if problem == "regression":
            model_name = st.selectbox("알고리즘", ["Linear Regression", "Random Forest"], index=1)
        else:
            model_name = st.selectbox("알고리즘", ["Logistic Regression", "Random Forest"], index=1)

        colp1, colp2 = st.columns(2)
        with colp1:
            test_size = st.slider("검증 비율(test_size)", 0.1, 0.4, 0.2, 0.05)
        with colp2:
            random_state = st.number_input("랜덤 시드", value=42, step=1)

        with st.expander("전처리 옵션", expanded=False):
            c1, c2, c3 = st.columns(3)
            with c1:
                num_imputer = st.selectbox("수치 결측 전략", ["median", "mean", "most_frequent", "constant"], index=0)
                scaling = st.selectbox("스케일링", ["standard", "minmax", "robust", "none"], index=0)
            with c2:
                cat_imputer = st.selectbox("범주 결측 전략", ["most_frequent", "constant"], index=0)
                onehot_drop = st.selectbox("OneHot drop", ["none", "first", "if_binary"], index=0)
            with c3:
                num_fill_value = st.number_input("수치 상수값", value=0.0, step=1.0, help="수치 결측=constant 선택 시 사용") if num_imputer == "constant" else None
                cat_fill_value = st.text_input("범주 상수값", value="missing", help="범주 결측=constant 선택 시 사용") if cat_imputer == "constant" else None

            apply_iqr = st.checkbox("IQR 이상치 제거(학습 전)", value=False)
            iqr_cols = []
            iqr_k = 1.5
            if apply_iqr:
                num_cols, _, _ = detect_column_types(df)
                iqr_cols = st.multiselect("이상치 제거 대상 컬럼(수치)", num_cols, default=num_cols)
                iqr_k = st.slider("IQR 배수(k)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

        st.markdown("---")
        use_tuning = st.checkbox("하이퍼파라미터 튜닝 사용", value=True)
        if use_tuning:
            search_type = st.selectbox("탐색 방법", ["Grid", "Random"], index=1)
            cv_folds = st.number_input("교차검증 폴드(CV)", value=5, min_value=2, max_value=10, step=1)
            if problem == "regression":
                scoring_options = {
                    "Auto (RMSE)": "neg_root_mean_squared_error",
                    "R2": "r2",
                    "MAE": "neg_mean_absolute_error",
                }
            else:
                scoring_options = {
                    "Auto (Accuracy)": "accuracy",
                    "F1(Weighted)": "f1_weighted",
                }
            scoring_label = st.selectbox("스코어링", list(scoring_options.keys()), index=0)
            scoring = scoring_options[scoring_label]
            n_iter = None
            if search_type == "Random":
                n_iter = st.number_input("Random Search 반복 수", value=25, min_value=5, max_value=200, step=5)

        submitted = st.form_submit_button("학습 실행")

    if not submitted:
        st.info("매개변수를 선택하고 '학습 실행'을 눌러주세요.")
        return

    with st.spinner("학습 중..."):
        try:
            work_df = df.copy()
            if 'apply_iqr' in locals() and apply_iqr and iqr_cols:
                work_df, stats_df = iqr_filter(work_df, iqr_cols, k=float(iqr_k))
                st.info(f"IQR 필터 적용: 제거된 행 수 {len(df) - len(work_df)}")

            preproc = {
                "num_imputer": num_imputer,
                "num_fill_value": num_fill_value,
                "cat_imputer": cat_imputer,
                "cat_fill_value": cat_fill_value,
                "scaling": None if scaling == "none" else scaling,
                "onehot_drop": None if onehot_drop == "none" else onehot_drop,
            }
            if use_tuning:
                model, metrics, figs, cv_results, best_params = tune_and_evaluate(
                    work_df,
                    target=target,
                    problem=problem,
                    model_name=model_name,
                    search="random" if search_type == "Random" else "grid",
                    scoring=scoring,
                    cv=int(cv_folds),
                    n_iter=int(n_iter) if n_iter else 25,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    preproc=preproc,
                )
            else:
                model, metrics, figs = train_and_evaluate(
                    work_df,
                    target=target,
                    problem=problem,
                    model_name=model_name,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    preproc=preproc,
                )
                cv_results, best_params = None, None
        except Exception as e:
            st.error(f"학습 실패: {e}")
            return

    st.markdown("#### 지표")
    m_items = list(metrics.items())
    cols = st.columns(len(m_items))
    for c, (k, v) in zip(cols, m_items):
        if isinstance(v, (int, float)):
            c.metric(k, f"{v:.4f}")
        else:
            c.metric(k, str(v))

    st.markdown("#### 진단 차트")
    if figs:
        for k, fig in figs.items():
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("표시할 차트가 없습니다.")

    if use_tuning and cv_results is not None:
        st.markdown("#### 교차검증 결과")
        colb1, colb2 = st.columns(2)
        with colb1:
            st.write("Best Params:")
            st.json(best_params)
        with colb2:
            best_score = cv_results.iloc[0]["mean_test_score"] if "mean_test_score" in cv_results.columns else None
            if best_score is not None:
                st.metric("Best CV Score", f"{best_score:.5f}")
        with st.expander("CV Results Table", expanded=False):
            st.dataframe(cv_results, use_container_width=True)

    st.markdown("#### 해석: 피처 중요도 / 계수")
    try:
        fi = extract_feature_importances(model)
        if not fi.empty:
            topn = st.slider("Top N", min_value=5, max_value=min(50, len(fi)), value=min(20, len(fi)))
            fig_imp = plot_feature_importance(fi, top_n=topn)
            st.plotly_chart(fig_imp, use_container_width=True)
            with st.expander("표 데이터 보기"):
                st.dataframe(fi.head(topn), use_container_width=True)
        else:
            st.info("이 모델에서는 중요도/계수를 제공하지 않습니다.")
    except Exception as e:
        st.warning(f"중요도 계산 실패: {e}")

    st.markdown("#### 모델 다운로드")
    try:
        blob = serialize_model_to_bytes(model)
        st.download_button(
            label="학습 파이프라인 저장(.pkl)",
            data=blob,
            file_name="model_pipeline.pkl",
            mime="application/octet-stream",
        )
    except Exception as e:
        st.warning(f"모델 직렬화 실패: {e}")

    st.markdown("#### 예측 / 내보내기")
    pred_source = st.radio("예측 데이터", ["현재 데이터셋", "CSV 업로드"], index=0, horizontal=True)
    pred_col = st.text_input("예측 컬럼명", value="prediction")
    include_original = st.checkbox("원본 컬럼 포함", value=True)
    include_proba = False
    model_step = getattr(model, "named_steps", {}).get("model", None)
    if model_step is not None and hasattr(model_step, "predict_proba"):
        include_proba = st.checkbox("분류 확률 포함(predict_proba)", value=False)

    pred_df = None
    if pred_source == "현재 데이터셋":
        Xf = df.drop(columns=[target]) if target in df.columns else df.copy()
        Xf = align_columns_for_inference(model, Xf)
        try:
            yhat = model.predict(Xf)
            out = df.copy() if include_original else Xf.copy()
            out[pred_col] = yhat
            if include_proba and hasattr(model_step, "predict_proba"):
                try:
                    proba = model_step.predict_proba(model.named_steps["preprocess"].transform(Xf))
                    classes = getattr(model_step, "classes_", None)
                    if classes is None:
                        classes = list(range(proba.shape[1]))
                    for i, cls in enumerate(classes):
                        out[f"proba_{cls}"] = proba[:, i]
                except Exception:
                    st.info("확률 계산을 진행할 수 없습니다.")
            pred_df = out
        except Exception as e:
            st.error(f"예측 실패: {e}")
    else:
        up = st.file_uploader("예측용 CSV 업로드", type=["csv"])
        c1, c2, c3 = st.columns(3)
        with c1:
            sep = st.text_input("구분자", value=",")
        with c2:
            decimal = st.text_input("소수점", value=".")
        with c3:
            encoding = st.text_input("인코딩", value="utf-8")
        drop_target = st.checkbox("업로드 데이터에 타깃 컬럼이 있으면 삭제", value=True)
        if up is not None:
            udf = load_csv(up, sep=sep or ",", decimal=decimal or ".", encoding=encoding or "utf-8")
            if drop_target and target in udf.columns:
                udf = udf.drop(columns=[target])
            Xf = align_columns_for_inference(model, udf)
            try:
                yhat = model.predict(Xf)
                out = udf.copy() if include_original else Xf.copy()
                out[pred_col] = yhat
                if include_proba and hasattr(model_step, "predict_proba"):
                    try:
                        proba = model_step.predict_proba(model.named_steps["preprocess"].transform(Xf))
                        classes = getattr(model_step, "classes_", None)
                        if classes is None:
                            classes = list(range(proba.shape[1]))
                        for i, cls in enumerate(classes):
                            out[f"proba_{cls}"] = proba[:, i]
                    except Exception:
                        st.info("확률 계산을 진행할 수 없습니다.")
                pred_df = out
            except Exception as e:
                st.error(f"예측 실패: {e}")

    if pred_df is not None and not pred_df.empty:
        with st.expander("예측 결과 미리보기", expanded=True):
            st.dataframe(pred_df.head(1000), use_container_width=True)
        csv_pred = pred_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            label="예측 결과 CSV 다운로드",
            data=csv_pred,
            file_name="predictions.csv",
            mime="text/csv",
        )


def main():
    st.title(f"📊 {APP_NAME}")
    st.caption(TAGLINE)

    df = sidebar_data_source()
    if df is None or df.empty:
        st.warning("좌측 사이드바에서 데이터를 불러오세요.")
        st.stop()

    tabs = st.tabs(["개요", "탐색", "시각화", "변환", "모델"])
    with tabs[0]:
        show_overview(df)
    with tabs[1]:
        show_explore(df)
    with tabs[2]:
        show_visualize(df)
    with tabs[3]:
        show_transform(df)
    with tabs[4]:
        show_model(df)


if __name__ == "__main__":
    main()
