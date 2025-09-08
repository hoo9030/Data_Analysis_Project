import importlib
m = importlib.import_module('backend.asgi_combined')
from data_ops import generate_sample_data

df = generate_sample_data(rows=200, seed=1)
# Violin
v = m._build_figure(df, chart='violin', x='category', y='feature_1', color=None, bins=30, log_x=False, log_y=False, facet_row=None, facet_col=None, norm=None, barmode=None)
print('violin_ok', isinstance(v, dict), len(v.get('data',[])))
# Scatter matrix
s = m._build_figure(df, chart='scatter_matrix', x=None, y=None, color='category', bins=4, log_x=False, log_y=False, facet_row=None, facet_col=None, norm=None, barmode=None)
print('splom_ok', isinstance(s, dict), len(s.get('data',[])))
# Feature importance
fi = m._build_figure(df, chart='feature_importance', x=None, y='target', color=None, bins=10, log_x=False, log_y=False, facet_row=None, facet_col=None, norm=None, barmode=None)
print('fi_ok', isinstance(fi, dict), len(fi.get('data',[])))
