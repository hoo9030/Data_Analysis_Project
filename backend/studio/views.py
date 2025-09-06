from django.shortcuts import render
from django.http import JsonResponse, HttpRequest
import io
import pandas as pd

# Reuse existing core logic from src
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eda_ops import basic_info  # type: ignore


def home(request: HttpRequest):
    return render(request, 'studio/home.html')


def eda_summary(request: HttpRequest):
    if request.method != 'POST':
        return JsonResponse({'error': 'POST a CSV file under field "file"'}, status=405)

    f = request.FILES.get('file')
    if not f:
        return JsonResponse({'error': 'file is required'}, status=400)

    sep = request.POST.get('sep') or ','
    decimal = request.POST.get('decimal') or '.'
    encoding = request.POST.get('encoding') or 'utf-8'

    try:
        # Read into memory and parse via pandas to avoid Streamlit dependency
        data = f.read()
        df = pd.read_csv(io.BytesIO(data), sep=sep, decimal=decimal, encoding=encoding)

        info = basic_info(df)
        # Ensure JSON serializable
        out = {
            'rows': int(info.get('rows', len(df))),
            'columns': int(info.get('columns', df.shape[1] if not df.empty else 0)),
            'memory': str(info.get('memory', '')),
            'missing': info.get('missing', {}),
        }
        return JsonResponse(out)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


# Create your views here.
