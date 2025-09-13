Extended APIs for Data Analysis Studio

- POST `/api/datasets/{id}/compute`
  - Body: `{ "expr": "colA + colB", "out_col": "(optional)", "inplace": false, "out_id": "(optional)" }`
  - Computes an expression using pandas eval and writes/overwrites `out_col`.

- POST `/api/datasets/{id}/groupby`
  - Body: `{ "by": ["col"], "aggs": { "target": ["mean","sum"] }, "dropna": true, "as_index": false, "out_id": "(optional)" }`
  - Group by columns and aggregate with common reducers.

- POST `/api/datasets/{id}/merge`
  - Body: `{ "right_id": "other", "on": ["key"], "how": "inner|left|right|outer", "suffixes": ["_x","_y"], "out_id": "(optional)" }`
  - Joins two datasets by keys.

- POST `/api/datasets/{id}/sort`
  - Body: `{ "by": ["-col1","col2"], "limit": 100, "out_id": "(optional)" }`
  - Sorts by columns (prefix `-` means descending) and optionally limits rows.

- POST `/api/datasets/{id}/dedup`
  - Body: `{ "subset": ["col1"], "keep": "first|last|none", "out_id": "(optional)" }`
  - Drops duplicate rows.

- GET `/api/datasets/{id}/profile?sample=10000`
  - Per-column profile: dtype, non_null, nulls, distinct, top value/count, numeric stats.

ML APIs

- POST `/api/ml/train`
  - Body: `{ "dataset_id": "id", "target": "y", "model": "linreg|logreg|rf_reg|rf_clf", "features": ["f1","f2"], "test_size": 0.2, "model_id": "(optional)", "sample_rows": 10000, "sample_frac": 0.5 }`
  - Trains model with preprocessing (numeric: impute+scale, categorical: impute+one-hot). Returns metrics and model_id.

- POST `/api/ml/{model_id}/predict_dataset`
  - Body: `{ "dataset_id": "id", "out_id": "(optional)", "proba": true }`
  - Runs prediction and writes a new dataset with `prediction` (and optionally `prediction_proba`).

- GET `/api/ml/models` → `{ items: [...], count }`
- DELETE `/api/ml/models/{id}` → delete a saved model.
