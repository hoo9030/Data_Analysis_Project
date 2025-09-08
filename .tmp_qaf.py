import importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()/ 'src'))
from data_ops import generate_sample_data
m = importlib.import_module('backend.asgi_combined')

df = generate_sample_data(rows=100, seed=123)
print('orig_shape', df.shape)

# Include cols
f1 = m._apply_filters(df, include_cols='feature_1,category', filter_query=None, limit_rows=None)
print('inc_cols', f1.shape, list(f1.columns))

# Query numeric and equality
f2 = m._apply_filters(df, include_cols=None, filter_query="feature_1 > 0 and category == 'A'", limit_rows=10)
print('query_limit', f2.shape, f2['category'].unique().tolist()[:3])

# Visualize after filter
spec = m._build_figure(f2, chart='histogram', x='feature_1', y=None, color='category', bins=20, log_x=False, log_y=False, facet_row=None, facet_col=None, norm=None, barmode=None)
print('fig_ok', isinstance(spec, dict), len(spec.get('data',[])))
