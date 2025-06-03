from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views import View
import json

# Basic webhook views for the multimodal content analyzer

@csrf_exempt
def webhook_test(request):
    """Simple test endpoint for webhooks"""
    if request.method == 'POST':
        return JsonResponse({'status': 'success', 'message': 'Webhook received'})
    return JsonResponse({'status': 'error', 'message': 'Only POST allowed'})

@method_decorator(csrf_exempt, name='dispatch')
class WebhookHandler(View):
    """Base webhook handler class"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            # Process webhook data here
            return JsonResponse({
                'status': 'success',
                'received': data
            })
        except json.JSONDecodeError:
            return JsonResponse({
                'status': 'error',
                'message': 'Invalid JSON'
            }, status=400)
    
    def get(self, request):
        return JsonResponse({
            'status': 'info',
            'message': 'Webhook endpoint ready'
        })