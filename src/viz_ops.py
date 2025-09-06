import pandas as pd
import plotly.express as px


def plot_correlation_heatmap(corr: pd.DataFrame):
    fig = px.imshow(
        corr,
        text_auto=False,
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_histogram(df: pd.DataFrame, x: str, color: str | None = None, nbins: int = 30):
    fig = px.histogram(df, x=x, color=color, nbins=nbins, opacity=0.85)
    fig.update_layout(bargap=0.05, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.scatter(df, x=x, y=y, color=color, opacity=0.8, trendline=None)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_box(df: pd.DataFrame, x: str | None, y: str, color: str | None = None):
    fig = px.box(df, x=x, y=y, color=color, points=False)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_bar_count(df: pd.DataFrame, x: str, color: str | None = None):
    tmp = df.copy()
    tmp["__count__"] = 1
    fig = px.bar(tmp, x=x, y="__count__", color=color, barmode="group")
    fig.update_yaxes(title="count")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_line(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_feature_importance(df: pd.DataFrame, feature_col: str = "feature", value_col: str = "importance", top_n: int = 30):
    d = df[[feature_col, value_col]].head(top_n).copy()
    d = d.iloc[::-1]  # reverse for horizontal bar top-down
    fig = px.bar(d, x=value_col, y=feature_col, orientation="h")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_scatter_matrix(df: pd.DataFrame, dimensions: list[str], color: str | None = None):
    fig = px.scatter_matrix(df, dimensions=dimensions, color=color, opacity=0.6)
    fig.update_traces(diagonal_visible=True)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_density_contour(df: pd.DataFrame, x: str, y: str, color: str | None = None):
    fig = px.density_contour(df, x=x, y=y, color=color, contours_coloring="fill")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_density_heatmap(df: pd.DataFrame, x: str, y: str, nbinsx: int = 30, nbinsy: int = 30):
    fig = px.density_heatmap(df, x=x, y=y, nbinsx=nbinsx, nbinsy=nbinsy, color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_histogram_facet(
    df: pd.DataFrame,
    x: str,
    facet_row: str | None = None,
    facet_col: str | None = None,
    color: str | None = None,
    nbins: int = 30,
):
    fig = px.histogram(df, x=x, color=color, nbins=nbins, facet_row=facet_row, facet_col=facet_col, opacity=0.85)
    fig.update_layout(bargap=0.05, margin=dict(l=10, r=10, t=30, b=10))
    return fig


def plot_violin_facet(
    df: pd.DataFrame,
    y: str,
    x: str | None = None,
    facet_row: str | None = None,
    facet_col: str | None = None,
    color: str | None = None,
    points: str | bool = "outliers",
):
    fig = px.violin(df, x=x, y=y, color=color, facet_row=facet_row, facet_col=facet_col, box=True, points=points)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig
