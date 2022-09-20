from django.urls import path
from .views import *

urlpatterns = [
    path('search/', Search.as_view()),
    path('create/', create),
    path('map/', MAP.as_view())
]
