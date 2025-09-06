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

