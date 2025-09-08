import importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()/ 'src'))
m = importlib.import_module('backend.asgi_combined')
from data_ops import generate_sample_data

df = generate_sample_data(rows=200, seed=2)
agg_df = m._apply_aggregation(df, group_by='category', agg='mean', value_cols='feature_1', pivot_col=None, pivot_fill=None)
print('agg1_shape', agg_df.shape)
agg_df2 = m._apply_aggregation(df, group_by='category', agg='mean', value_cols='feature_1', pivot_col='flag', pivot_fill=0)
print('agg2_shape', agg_df2.shape, 'cols', list(agg_df2.columns)[:5])
