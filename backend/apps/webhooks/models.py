from django.db import models
from django.contrib.auth import get_user_model
import uuid

User = get_user_model()

class WebhookEvent(models.TextChoices):
    ANALYSIS_STARTED = 'analysis.started', 'Analysis Started'
    ANALYSIS_COMPLETED = 'analysis.completed', 'Analysis Completed'
    ANALYSIS_FAILED = 'analysis.failed', 'Analysis Failed'
    CONTENT_UPLOADED = 'content.uploaded', 'Content Uploaded'
    MULTIMODAL_COMPLETED = 'multimodal.completed', 'Multimodal Analysis Completed'

class Webhook(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='webhooks')
    
    # Webhook configuration
    name = models.CharField(max_length=255)
    url = models.URLField()
    secret = models.CharField(max_length=255, blank=True)
    events = models.JSONField(default=list)  # List of WebhookEvent values
    
    # Settings
    is_active = models.BooleanField(default=True)
    retry_count = models.IntegerField(default=3)
    timeout = models.IntegerField(default=30)  # seconds
    
    # Statistics
    total_deliveries = models.IntegerField(default=0)
    successful_deliveries = models.IntegerField(default=0)
    failed_deliveries = models.IntegerField(default=0)
    last_delivery_at = models.DateTimeField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'webhooks'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} - {self.url}"

class WebhookDelivery(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    webhook = models.ForeignKey(Webhook, on_delete=models.CASCADE, related_name='deliveries')
    
    # Event information
    event = models.CharField(max_length=50, choices=WebhookEvent.choices)
    payload = models.JSONField(default=dict)
    
    # Delivery information
    status_code = models.IntegerField(blank=True, null=True)
    response_body = models.TextField(blank=True)
    error_message = models.TextField(blank=True)
    delivery_time = models.DurationField(blank=True, null=True)
    
    # Retry information
    attempt_count = models.IntegerField(default=1)
    is_successful = models.BooleanField(default=False)
    next_retry_at = models.DateTimeField(blank=True, null=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    delivered_at = models.DateTimeField(blank=True, null=True)

    class Meta:
        db_table = 'webhook_deliveries'
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.webhook.name} - {self.event} - {self.created_at}"
