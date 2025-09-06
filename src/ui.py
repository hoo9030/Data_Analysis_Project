import streamlit as st

from .settings import APP_NAME, TAGLINE
from .styles import apply_base_theme


def render_header(icon: str = "📊", show_tagline: bool = True) -> None:
    """Studio header with title and optional tagline."""
    apply_base_theme()
    title_html = f"""
    <div class="studio-header">
      <div class="studio-title">{icon} {APP_NAME}</div>
      {f'<div class="studio-tagline">{TAGLINE}</div>' if show_tagline and TAGLINE else ''}
    </div>
    """
    st.markdown(title_html, unsafe_allow_html=True)


def divider() -> None:
    st.markdown("<hr style='border: 1px solid #e5e7eb; margin: 0.8rem 0;' />", unsafe_allow_html=True)

