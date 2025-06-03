from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver
# You can import your content models here when they're defined
# from .models import Content

# Content-related signals for multimodal content analyzer

@receiver(post_save)
def content_saved_signal(sender, instance, created, **kwargs):
    """
    Signal fired when content is saved
    """
    if created:
        print(f"New content created: {instance}")
    else:
        print(f"Content updated: {instance}")

@receiver(post_delete)
def content_deleted_signal(sender, instance, **kwargs):
    """
    Signal fired when content is deleted
    """
    print(f"Content deleted: {instance}")