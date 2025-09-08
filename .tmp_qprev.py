import importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()/ 'src'))
m = importlib.import_module('backend.asgi_combined')
from data_ops import generate_sample_data

df = generate_sample_data(rows=5, seed=7)
prev = m._preview_records(df, max_rows=3)
print('preview_len', len(prev), 'keys', list(prev[0].keys()))
