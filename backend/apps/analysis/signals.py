from django.db.models.signals import post_save
from django.dispatch import receiver
# You can import your analysis models here when they're defined
# from .models import Analysis

# Analysis-related signals for multimodal content analyzer

@receiver(post_save)
def analysis_completed_signal(sender, instance, created, **kwargs):
    """
    Signal fired when analysis is completed
    """
    if created:
        print(f"New analysis created: {instance}")
    else:
        print(f"Analysis updated: {instance}")