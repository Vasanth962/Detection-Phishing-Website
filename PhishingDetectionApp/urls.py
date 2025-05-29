from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="home"),
    path("index.html", views.index, name="index"),
    path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
    path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
    path("RunLGBM", views.RunLGBM, name="RunLGBM"),
    path("RunSVM", views.RunSVM, name="RunSVM"),
    path("Predict", views.Predict, name="Predict"),
    path("PredictAction", views.PredictAction, name="PredictAction"),
]
