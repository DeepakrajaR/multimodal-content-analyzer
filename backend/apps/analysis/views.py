from rest_framework import generics, status, permissions
from rest_framework.decorators import api_view, permission_classes
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import AnalysisJob, AnalysisResult, MultimodalAnalysis
from .serializers import (
    AnalysisJobSerializer, AnalysisJobCreateSerializer,
    AnalysisResultSerializer, MultimodalAnalysisSerializer
)
from .tasks import start_analysis_job

class AnalysisJobListCreateView(generics.ListCreateAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def get_serializer_class(self):
        if self.request.method == 'POST':
            return AnalysisJobCreateSerializer
        return AnalysisJobSerializer

    def get_queryset(self):
        queryset = AnalysisJob.objects.filter(user=self.request.user)
        
        # Filter by status
        status_filter = self.request.query_params.get('status', None)
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        # Filter by analysis types
        analysis_type = self.request.query_params.get('analysis_type', None)
        if analysis_type:
            queryset = queryset.filter(analysis_types__contains=[analysis_type])
        
        return queryset.order_by('-created_at')

    def perform_create(self, serializer):
        job = serializer.save()
        # Start the analysis job asynchronously
        start_analysis_job.delay(job.id)

class AnalysisJobDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = AnalysisJobSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return AnalysisJob.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def cancel_analysis_job_view(request, job_id):
    job = get_object_or_404(
        AnalysisJob,
        id=job_id,
        user=request.user
    )
    
    if job.status in ['pending', 'running']:
        job.status = 'cancelled'
        job.save()
        
        # TODO: Cancel the actual background task
        
        return Response({
            'message': 'Analysis job cancelled successfully',
            'job': AnalysisJobSerializer(job).data
        })
    else:
        return Response({
            'error': f'Cannot cancel job with status: {job.status}'
        }, status=status.HTTP_400_BAD_REQUEST)

class AnalysisResultListView(generics.ListAPIView):
    serializer_class = AnalysisResultSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        job_id = self.kwargs.get('job_id')
        job = get_object_or_404(
            AnalysisJob,
            id=job_id,
            user=self.request.user
        )
        
        queryset = AnalysisResult.objects.filter(job=job)
        
        # Filter by analysis type
        analysis_type = self.request.query_params.get('analysis_type', None)
        if analysis_type:
            queryset = queryset.filter(analysis_type=analysis_type)
        
        return queryset.order_by('-created_at')

@api_view(['GET'])
@permission_classes([permissions.IsAuthenticated])
def analysis_stats_view(request):
    user = request.user
    
    stats = {
        'total_jobs': AnalysisJob.objects.filter(user=user).count(),
        'jobs_by_status': {},
        'popular_analysis_types': {},
        'total_processing_time': 0,
        'recent_jobs': [],
    }
    
    # Count by status
    for status_choice, _ in AnalysisJob._meta.get_field('status').choices:
        count = AnalysisJob.objects.filter(
            user=user, 
            status=status_choice
        ).count()
        stats['jobs_by_status'][status_choice] = count
    
    # Get popular analysis types
    jobs = AnalysisJob.objects.filter(user=user)
    analysis_type_counts = {}
    for job in jobs:
        for analysis_type in job.analysis_types:
            analysis_type_counts[analysis_type] = analysis_type_counts.get(analysis_type, 0) + 1
    
    stats['popular_analysis_types'] = dict(
        sorted(analysis_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    )
    
    # Recent jobs
    recent_jobs = AnalysisJob.objects.filter(user=user).order_by('-created_at')[:5]
    stats['recent_jobs'] = AnalysisJobSerializer(recent_jobs, many=True).data
    
    return Response(stats)

class MultimodalAnalysisListCreateView(generics.ListCreateAPIView):
    serializer_class = MultimodalAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MultimodalAnalysis.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        multimodal_analysis = serializer.save()
        # TODO: Start multimodal fusion task
        # start_multimodal_fusion.delay(multimodal_analysis.id)

class MultimodalAnalysisDetailView(generics.RetrieveUpdateDestroyAPIView):
    serializer_class = MultimodalAnalysisSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return MultimodalAnalysis.objects.filter(user=self.request.user)

@api_view(['POST'])
@permission_classes([permissions.IsAuthenticated])
def quick_analyze_view(request):
    """Quick analysis endpoint for simple text/image analysis"""
    analysis_type = request.data.get('analysis_type')
    content = request.data.get('content')  # text content or base64 image
    content_type = request.data.get('content_type', 'text')  # text or image
    
    if not analysis_type or not content:
        return Response({
            'error': 'analysis_type and content are required'
        }, status=status.HTTP_400_BAD_REQUEST)
    
    # TODO: Implement quick analysis logic
    # For now, return a mock response
    return Response({
        'analysis_type': analysis_type,
        'content_type': content_type,
        'results': {
            'status': 'completed',
            'confidence': 0.85,
            'analysis': f'Mock {analysis_type} analysis result'
        },
        'processing_time': '0.5s'
    })
