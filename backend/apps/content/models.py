from django.db import models
from django.contrib.auth import get_user_model
import uuid
import os

User = get_user_model()

class ContentType(models.TextChoices):
    TEXT = 'text', 'Text'
    IMAGE = 'image', 'Image'
    VIDEO = 'video', 'Video'
    DOCUMENT = 'document', 'Document'
    AUDIO = 'audio', 'Audio'

class ProcessingStatus(models.TextChoices):
    PENDING = 'pending', 'Pending'
    PROCESSING = 'processing', 'Processing'
    COMPLETED = 'completed', 'Completed'
    FAILED = 'failed', 'Failed'
    CANCELLED = 'cancelled', 'Cancelled'

def upload_to(instance, filename):
    ext = filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    return os.path.join('uploads', str(instance.user.id), filename)

class ContentItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='content_items')
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    content_type = models.CharField(max_length=20, choices=ContentType.choices)
    
    # File information
    file = models.FileField(upload_to=upload_to, blank=True, null=True)
    file_size = models.BigIntegerField(default=0)
    file_name = models.CharField(max_length=255, blank=True)
    mime_type = models.CharField(max_length=100, blank=True)
    
    # Text content (for direct text input)
    text_content = models.TextField(blank=True)
    
    # Metadata
    metadata = models.JSONField(default=dict)
    tags = models.JSONField(default=list)
    
    # Processing information
    processing_status = models.CharField(
        max_length=20, 
        choices=ProcessingStatus.choices, 
        default=ProcessingStatus.PENDING
    )
    processing_progress = models.IntegerField(default=0)  # 0-100
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    processed_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'content_items'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} ({self.content_type})"

class ContentCollection(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='collections')
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    content_items = models.ManyToManyField(ContentItem, related_name='collections')
    tags = models.JSONField(default=list)
    is_public = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'content_collections'
        ordering = ['-created_at']

    def __str__(self):
        return self.name
