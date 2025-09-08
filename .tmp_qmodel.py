import importlib, sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd()/ 'src'))
m = importlib.import_module('backend.asgi_combined')
from data_ops import generate_sample_data

df = generate_sample_data(rows=200, seed=1)
# smoke: build pipeline and compute metrics
pipe, X, y = m._build_model_pipeline(df, 'target', 'regression')
print('Xshape', X.shape, 'ylen', len(y))
from sklearn.model_selection import KFold, cross_val_score
rmse = -cross_val_score(pipe, X, y, cv=KFold(n_splits=3, shuffle=True, random_state=42), scoring='neg_root_mean_squared_error')
print('rmse', rmse.mean())
