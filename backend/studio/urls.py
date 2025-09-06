from django.urls import path
from . import views


urlpatterns = [
    path('', views.home, name='home'),
    path('eda/summary', views.eda_summary, name='eda_summary'),
]

