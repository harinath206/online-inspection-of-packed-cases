from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),  # The upload image view
    path('submit_feedback/', views.submit_feedback, name='submit_feedback'),  # The feedback submission view
]
