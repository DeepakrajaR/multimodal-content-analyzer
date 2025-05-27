from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from django.db.models import Q
from .models import ContentItem, ContentCollection
from .serializers import (
    ContentItemSerializer, ContentItemCreateSerializer,
    ContentCollectionSerializer, ContentCollectionCreateSerializer
)

class ContentItemListCreateView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return ContentItemCreateSerializer
        return ContentItemSerializer

    def get_queryset(self):
        queryset = ContentItem.objects.filter(user=self.request.user)
        
        # Filter by content type
        content_type = self.request.query_params.get('content_type', None)
        if content_type:
            queryset = queryset.filter(content_type=content_type)
        
        # Filter by processing status
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(processing_status=status_filter)
        
        # Search by title or description
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(title__icontains=search) | Q(description__icontains=search)
            )
        
        # Filter by tags
        tags = self.request.query_params.get('tags', None)
        if tags:
            tag_list = tags.split(',')
            for tag in tag_list:
                queryset = queryset.filter(tags__contains=[tag.strip()])
        
        return queryset.order_by('-created_at')

class ContentItemDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = ContentItemSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ContentItem.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def bulk_upload_view(request):
    files = request.FILES.getlist('files')
    if not files:
        return Response({
            'error': 'No files provided'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    created_items = []
    errors = []
    
    for file in files:
        try:
            # Determine content type based on file
            content_type = 'document'  # default
            if file.content_type.startswith('image/'):
                content_type = 'image'
            elif file.content_type.startswith('video/'):
                content_type = 'video'
            elif file.content_type.startswith('audio/'):
                content_type = 'audio'
            
            content_item = ContentItem.objects.create(
                user=request.user,
                title=file.name,
                content_type=content_type,
                file=file,
                file_size=file.size,
                file_name=file.name,
                mime_type=file.content_type
            )
            created_items.append(ContentItemSerializer(content_item).data)
        except Exception as e:
            errors.append({
                'file': file.name,
                'error': str(e)
            })
    
    return Response({
        'created_items': created_items,
        'errors': errors,
        'total_created': len(created_items),
        'total_errors': len(errors)
    })

class ContentCollectionListCreateView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return ContentCollectionCreateSerializer
        return ContentCollectionSerializer

    def get_queryset(self):
        queryset = ContentCollection.objects.filter(user=self.request.user)
        
        # Search by name or description
        search = self.request.query_params.get('search', None)
        if search:
            queryset = queryset.filter(
                Q(name__icontains=search) | Q(description__icontains=search)
            )
        
        return queryset.order_by('-created_at')

class ContentCollectionDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = ContentCollectionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return ContentCollection.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def add_items_to_collection_view(request, collection_id):
    try:
        collection = ContentCollection.objects.get(
            id=collection_id, 
            user=request.user
        )
    except ContentCollection.DoesNotExist:
        return Response({
            'error': 'Collection not found'
        }, status=status.HTTP_404_NOT_FOUND)
    
    item_ids = request.data.get('item_ids', [])
    if not item_ids:
        return Response({
            'error': 'No item IDs provided'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # Get valid content items owned by the user
    items = ContentItem.objects.filter(
        id__in=item_ids,
        user=request.user
    )
    
    collection.content_items.add(*items)
    
    return Response({
        'message': f'Added {items.count()} items to collection',
        'collection': ContentCollectionSerializer(collection).data
    })

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def content_stats_view(request):
    user = request.user
    
    stats = {
        'total_items': ContentItem.objects.filter(user=user).count(),
        'items_by_type': {},
        'items_by_status': {},
        'total_collections': ContentCollection.objects.filter(user=user).count(),
        'storage_used': 0,
    }
    
    # Count by content type
    for content_type, _ in ContentItem.content_type.field.choices:
        count = ContentItem.objects.filter(
            user=user, 
            content_type=content_type
        ).count()
        stats['items_by_type'][content_type] = count
    
    # Count by processing status
    for status_choice, _ in ContentItem.processing_status.field.choices:
        count = ContentItem.objects.filter(
            user=user, 
            processing_status=status_choice
        ).count()
        stats['items_by_status'][status_choice] = count
    
    # Calculate total storage used
    items = ContentItem.objects.filter(user=user)
    stats['storage_used'] = sum(item.file_size for item in items if item.file_size)
    
    return Response(stats)
