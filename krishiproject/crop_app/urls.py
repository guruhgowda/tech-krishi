"""krishiproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('',views.index,name='index'),
    path('register',views.register,name="register"),

    path('homepage', views.homepage,name='homepage'),
    path('croppredict', views.croppredict,name='croppredict'),
    path('pestdetect', views.pestdetect,name='pestdetect'),
    path('weatherpredict', views.weatherpredict,name='weatherpredict'),
    path('rentingtools', views.rentingtools,name='rentingtools'),
    path('exprertadvice', views.exprertadvice,name='exprertadvice'),
    path('Login',views.Login,name="Login"),

]
