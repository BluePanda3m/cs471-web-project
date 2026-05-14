from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),  
    path('formulation/', views.formulation, name='formulation'),
    path('algorithm/', views.algorithm_view, name='algorithm'),
    path('results/', views.results, name='results'),
    path('discussion/', views.discussion, name='discussion'),
]