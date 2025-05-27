from celery import shared_task
from django.utils import timezone
from .models import AnalysisJob, AnalysisResult
import logging
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

@shared_task
def start_analysis_job(job_id):
    \"\"\"Start an analysis job and process it through AI services\"\"\"
    try:
        job = AnalysisJob.objects.get(id=job_id)
        job.status = 'running'
        job.started_at = timezone.now()
        job.progress = 0
        job.save()
        
        logger.info(f"Starting analysis job {job_id}")
        
        # Process each analysis type
        for analysis_type in job.analysis_types:
            try:
                result = process_analysis_type(job, analysis_type)
                if result:
                    # Save the result
                    AnalysisResult.objects.create(
                        job=job,
                        analysis_type=analysis_type,
                        results=result.get('results', {}),
                        confidence_score=result.get('confidence', None),
                        model_version=result.get('model_version', ''),
                        service_version=result.get('service_version', '')
                    )
                
                # Update progress
                progress = (job.results.count() / len(job.analysis_types)) * 100
                job.progress = int(progress)
                job.save()
                
            except Exception as e:
                logger.error(f"Error processing {analysis_type} for job {job_id}: {str(e)}")
                continue
        
        # Mark job as completed
        job.status = 'completed'
        job.completed_at = timezone.now()
        job.processing_time = job.completed_at - job.started_at
        job.progress = 100
        job.save()
        
        logger.info(f"Completed analysis job {job_id}")
        
    except AnalysisJob.DoesNotExist:
        logger.error(f"Analysis job {job_id} not found")
    except Exception as e:
        logger.error(f"Error in analysis job {job_id}: {str(e)}")
        # Mark job as failed
        try:
            job = AnalysisJob.objects.get(id=job_id)
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
        except:
            pass

def process_analysis_type(job, analysis_type):
    \"\"\"Process a specific analysis type\"\"\"
    
    # Determine which AI service to use
    service_url = None
    
    text_analyses = ['sentiment', 'entities', 'topics', 'summary', 'emotions', 'classification']
    image_analyses = ['objects', 'scenes', 'ocr', 'faces']
    video_analyses = ['transcription']
    
    if analysis_type in text_analyses:
        service_url = settings.AI_SERVICES['TEXT_ANALYZER_URL']
    elif analysis_type in image_analyses:
        service_url = settings.AI_SERVICES['IMAGE_ANALYZER_URL']
    elif analysis_type in video_analyses:
        service_url = settings.AI_SERVICES['VIDEO_ANALYZER_URL']
    
    if not service_url:
        logger.error(f"No service found for analysis type: {analysis_type}")
        return None
    
    try:
        # Prepare request data
        request_data = {
            'analysis_type': analysis_type,
            'configuration': job.configuration
        }
        
        # Add content information
        if job.content_item:
            if job.content_item.text_content:
                request_data['text_content'] = job.content_item.text_content
            if job.content_item.file:
                request_data['file_url'] = job.content_item.file.url
                request_data['content_type'] = job.content_item.content_type
        
        # Make request to AI service
        response = requests.post(
            f"{service_url}/analyze",
            json=request_data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"AI service error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error to AI service: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return None

@shared_task
def cleanup_old_jobs():
    \"\"\"Clean up old completed/failed jobs\"\"\"
    from datetime import timedelta
    
    # Delete jobs older than 30 days
    cutoff_date = timezone.now() - timedelta(days=30)
    old_jobs = AnalysisJob.objects.filter(
        created_at__lt=cutoff_date,
        status__in=['completed', 'failed', 'cancelled']
    )
    
    count = old_jobs.count()
    old_jobs.delete()
    
    logger.info(f"Cleaned up {count} old analysis jobs")
    return count
