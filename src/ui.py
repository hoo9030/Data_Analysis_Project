import streamlit as st

from .settings import APP_NAME, TAGLINE
from .styles import apply_base_theme


def render_topnav(active: str = "Studio") -> None:
    """Top navigation bar using page links.

    active: one of {"Home", "Studio", "Profiling"}
    """
    apply_base_theme()
    c1, c2, c3, csp = st.columns([1, 1, 1, 6])
    with c1:
        if active == "Home":
            st.markdown("<b>🏠 Home</b>", unsafe_allow_html=True)
        else:
            st.page_link("pages/1_Home.py", label="🏠 Home")
    with c2:
        if active == "Studio":
            st.markdown("<b>📊 Studio</b>", unsafe_allow_html=True)
        else:
            st.page_link("app.py", label="📊 Studio")
    with c3:
        if active == "Profiling":
            st.markdown("<b>📑 Profiling</b>", unsafe_allow_html=True)
        else:
            st.page_link("pages/5_Profiling.py", label="📑 Profiling")
    with csp:
        pass


def render_footer() -> None:
    apply_base_theme()
    st.markdown(
        """
        <div style="margin-top:1.2rem;padding:0.8rem 0;color:#6B7280;border-top:1px solid #e5e7eb;">
          <div>© Studio — Built with Streamlit</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
