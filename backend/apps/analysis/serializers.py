from rest_framework import serializers
from .models import AnalysisJob, AnalysisResult, MultimodalAnalysis
from apps.content.models import ContentItem, ContentCollection

class AnalysisJobSerializer(serializers.ModelSerializer):
    results = serializers.SerializerMethodField()
    content_item_title = serializers.SerializerMethodField()
    collection_name = serializers.SerializerMethodField()

    class Meta:
        model = AnalysisJob
        fields = '__all__'
        read_only_fields = ('id', 'user', 'status', 'progress', 'error_message',
                           'started_at', 'completed_at', 'processing_time',
                           'created_at', 'updated_at')

    def get_results(self, obj):
        results = AnalysisResult.objects.filter(job=obj)
        return AnalysisResultSerializer(results, many=True).data

    def get_content_item_title(self, obj):
        return obj.content_item.title if obj.content_item else None

    def get_collection_name(self, obj):
        return obj.content_collection.name if obj.content_collection else None

class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisJob
        fields = ('content_item', 'content_collection', 'analysis_types', 'configuration')

    def validate(self, attrs):
        if not attrs.get('content_item') and not attrs.get('content_collection'):
            raise serializers.ValidationError(
                "Either content_item or content_collection must be provided"
            )
        
        if attrs.get('content_item') and attrs.get('content_collection'):
            raise serializers.ValidationError(
                "Cannot specify both content_item and content_collection"
            )
        
        return attrs

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)

class AnalysisResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalysisResult
        fields = '__all__'

class MultimodalAnalysisSerializer(serializers.ModelSerializer):
    analysis_jobs_details = serializers.SerializerMethodField()

    class Meta:
        model = MultimodalAnalysis
        fields = '__all__'
        read_only_fields = ('id', 'user', 'status', 'created_at', 'updated_at', 'completed_at')

    def get_analysis_jobs_details(self, obj):
        jobs = obj.analysis_jobs.all()
        return AnalysisJobSerializer(jobs, many=True).data

    def create(self, validated_data):
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)
