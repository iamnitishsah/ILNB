from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import Bot_Response, Trade_News

# URL pattern for chatbot app
urlpatterns = [
    path('chat/', Bot_Response.as_view(), name='chat'),
    path('news/', Trade_News.as_view(), name='stock_news'),
]