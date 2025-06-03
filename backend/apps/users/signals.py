from django.db.models.signals import post_save, pre_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from django.db import models

# Custom signals for user management in multimodal content analyzer

@receiver(post_save, sender=User)
def user_created_signal(sender, instance, created, **kwargs):
    """
    Signal fired when a user is created or updated
    """
    if created:
        # Log user creation
        print(f"New user created: {instance.username} ({instance.email})")
        
        # You can add additional logic here such as:
        # - Creating user profile
        # - Setting default preferences
        # - Sending welcome email
        # - Creating user-specific directories
        
    else:
        # User was updated
        print(f"User updated: {instance.username}")

@receiver(pre_save, sender=User)
def user_pre_save_signal(sender, instance, **kwargs):
    """
    Signal fired before a user is saved
    """
    # You can add pre-save validation or modifications here
    pass

# You can add more custom signals here as needed for your multimodal content analyzer
# For example:
# - Content upload signals
# - Analysis completion signals
# - User activity tracking signals