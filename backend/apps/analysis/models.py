from django.db import models
from django.contrib.auth import get_user_model
from apps.content.models import ContentItem, ContentCollection
import uuid

User = get_user_model()

class AnalysisType(models.TextChoices):
    SENTIMENT = 'sentiment', 'Sentiment Analysis'
    ENTITIES = 'entities', 'Entity Extraction'
    TOPICS = 'topics', 'Topic Modeling'
    SUMMARY = 'summary', 'Text Summarization'
    OBJECTS = 'objects', 'Object Detection'
    SCENES = 'scenes', 'Scene Classification'
    OCR = 'ocr', 'Optical Character Recognition'
    FACES = 'faces', 'Face Detection'
    TRANSCRIPTION = 'transcription', 'Audio Transcription'
    EMOTIONS = 'emotions', 'Emotion Detection'
    CLASSIFICATION = 'classification', 'Content Classification'
    SIMILARITY = 'similarity', 'Similarity Analysis'

class AnalysisJob(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analysis_jobs')
    
    # Content to analyze
    content_item = models.ForeignKey(
        ContentItem, 
        on_delete=models.CASCADE, 
        related_name='analysis_jobs',
        blank=True, 
        null=True
    )
    content_collection = models.ForeignKey(
        ContentCollection,
        on_delete=models.CASCADE,
        related_name='analysis_jobs',
        blank=True,
        null=True
    )
    
    # Analysis configuration
    analysis_types = models.JSONField(default=list)  # List of AnalysisType values
    configuration = models.JSONField(default=dict)  # Analysis-specific config
    
    # Job status
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('running', 'Running'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
            ('cancelled', 'Cancelled'),
        ],
        default='pending'
    )
    progress = models.IntegerField(default=0)  # 0-100
    error_message = models.TextField(blank=True)
    
    # Processing information
    started_at = models.DateTimeField(blank=True, null=True)
    completed_at = models.DateTimeField(blank=True, null=True)
    processing_time = models.DurationField(blank=True, null=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'analysis_jobs'
        ordering = ['-created_at']

    def __str__(self):
        return f"Analysis Job {self.id} - {self.status}"

class AnalysisResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job = models.ForeignKey(AnalysisJob, on_delete=models.CASCADE, related_name='results')
    analysis_type = models.CharField(max_length=20, choices=AnalysisType.choices)
    
    # Results data
    results = models.JSONField(default=dict)
    confidence_score = models.FloatField(blank=True, null=True)
    processing_time = models.DurationField(blank=True, null=True)
    
    # Metadata
    model_version = models.CharField(max_length=100, blank=True)
    service_version = models.CharField(max_length=100, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'analysis_results'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.analysis_type} - {self.job.id}"

class MultimodalAnalysis(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='multimodal_analyses')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Input jobs
    analysis_jobs = models.ManyToManyField(AnalysisJob, related_name='multimodal_analyses')
    
    # Fusion results
    fusion_results = models.JSONField(default=dict)
    insights = models.JSONField(default=dict)
    correlations = models.JSONField(default=dict)
    
    # Status
    status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed'),
        ],
        default='pending'
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'multimodal_analyses'
        ordering = ['-created_at']

    def __str__(self):
        return self.name
