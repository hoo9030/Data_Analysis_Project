import streamlit as st


_CSS = r"""
/* Studio base theme — typography, spacing, components */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

:root {
  --brand-primary: #4F46E5; /* Indigo 600 */
  --brand-secondary: #06B6D4; /* Cyan 500 */
  --text-strong: #111827; /* Gray 900 */
  --text-muted: #6B7280; /* Gray 500 */
  --surface-1: #FFFFFF;
  --surface-2: #F6F8FC;
  --border: #E5E7EB; /* Gray 200 */
  --radius: 12px;
}

html, body, [data-testid="stAppViewContainer"] * {
  font-family: 'Inter', 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic',
               'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important;
}

/* Sidebar polish */
[data-testid="stSidebar"] {
  background: var(--surface-2) !important;
}

/* Reduce top padding, widen content feel */
section.main > div.block-container {
  padding-top: 1.2rem;
}

/* Studio header */
.studio-header {
  border: 1px solid var(--border);
  background: linear-gradient(180deg, #ffffff 0%, #f7f8fd 100%);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
  margin: 0 0 1.2rem 0;
}
.studio-title {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  font-weight: 700;
  font-size: 1.15rem;
  color: var(--text-strong);
}
.studio-tagline {
  margin-top: 0.1rem;
  color: var(--text-muted);
  font-size: 0.95rem;
}

/* Metric cards */
div[data-testid="stMetric"] {
  border: 1px solid var(--border);
  background: var(--surface-2);
  border-radius: var(--radius);
  padding: 0.75rem 0.9rem;
}

/* Hide deploy button in toolbar for a cleaner look (optional) */
[data-testid="Toolbar"] > div:nth-child(2) {
  display: none;
}
"""


def apply_base_theme() -> None:
    """Inject Studio CSS once per session."""
    if st.session_state.get("_studio_css_loaded"):
        return
    st.markdown(f"""
        <style>
        {_CSS}
        </style>
    """, unsafe_allow_html=True)
    st.session_state["_studio_css_loaded"] = True

