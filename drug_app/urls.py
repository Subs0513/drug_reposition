from django.urls import path
from . import views

urlpatterns = [
    path('search/', views.drug_search, name='drug_search'),
]