import importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()/ 'src'))
m = importlib.import_module('backend.asgi_combined')
from data_ops import generate_sample_data

df = generate_sample_data(rows=200, seed=2)
# Apply aggregation: group by category, mean of feature_1
agg_df = m._apply_aggregation(df, group_by='category', agg='mean', value_cols='feature_1', pivot_col=None, pivot_fill=None)
print('agg1_shape', agg_df.shape)
# Pivot: index=category, columns=flag, values=feature_1 mean
agg_df2 = m._apply_aggregation(df, group_by='category', agg='mean', value_cols='feature_1', pivot_col='flag', pivot_fill=0)
print('agg2_shape', agg_df2.shape, 'cols', list(agg_df2.columns)[:5])
# Use chart bar_value with aggregated df
spec = m._build_figure(agg_df, chart='bar_value', x='category', y='feature_1', color=None, bins=0, log_x=False, log_y=False, facet_row=None, facet_col=None, norm=None, barmode=None)
print('bar_value_ok', isinstance(spec, dict), len(spec.get('data',[])))
