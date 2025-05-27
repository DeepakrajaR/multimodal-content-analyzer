from django.urls import path
from . import views

urlpatterns = [
    # Content Items
    path('items/', views.ContentItemListCreateView.as_view(), name='content-items'),
    path('items/<uuid:pk>/', views.ContentItemDetailView.as_view(), name='content-item-detail'),
    path('bulk-upload/', views.bulk_upload_view, name='bulk-upload'),
    path('stats/', views.content_stats_view, name='content-stats'),
    
    # Collections
    path('collections/', views.ContentCollectionListCreateView.as_view(), name='collections'),
    path('collections/<uuid:pk>/', views.ContentCollectionDetailView.as_view(), name='collection-detail'),
    path('collections/<uuid:collection_id>/add-items/', views.add_items_to_collection_view, name='add-items-to-collection'),
]
