from django.urls import path
from .views import Bot_Response

urlpatterns = [
    path('chat/', Bot_Response.as_view(), name='chat'),
]