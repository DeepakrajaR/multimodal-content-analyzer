from django.urls import path
from . import views

urlpatterns = [
    # Analysis Jobs
    path('jobs/', views.AnalysisJobListCreateView.as_view(), name='analysis-jobs'),
    path('jobs/<uuid:pk>/', views.AnalysisJobDetailView.as_view(), name='analysis-job-detail'),
    path('jobs/<uuid:job_id>/cancel/', views.cancel_analysis_job_view, name='cancel-analysis-job'),
    path('jobs/<uuid:job_id>/results/', views.AnalysisResultListView.as_view(), name='analysis-results'),
    
    # Quick Analysis
    path('quick/', views.quick_analyze_view, name='quick-analyze'),
    
    # Statistics
    path('stats/', views.analysis_stats_view, name='analysis-stats'),
    
    # Multimodal Analysis
    path('multimodal/', views.MultimodalAnalysisListCreateView.as_view(), name='multimodal-analyses'),
    path('multimodal/<uuid:pk>/', views.MultimodalAnalysisDetailView.as_view(), name='multimodal-analysis-detail'),
]
