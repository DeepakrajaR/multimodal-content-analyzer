from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import tempfile
import shutil
from loguru import logger
import asyncio

from analyzer import VideoAnalyzer

app = FastAPI(
    title="Video Analyzer Service",
    description="Advanced video processing and analysis service for multimodal content analyzer",
    version="1.0.0"
)

# Initialize the analyzer
analyzer = VideoAnalyzer()

class AnalysisRequest(BaseModel):
    analysis_type: str
    configuration: Optional[Dict[str, Any]] = {}
    file_url: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: float
    model_version: str
    service_version: str = "1.0.0"

class BatchAnalysisRequest(BaseModel):
    analysis_types: List[str]
    configuration: Optional[Dict[str, Any]] = {}

class VideoProcessingJob(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# In-memory job storage (in production, use Redis or database)
processing_jobs = {}

@app.get("/")
async def root():
    return {
        "service": "Video Analyzer",
        "version": "1.0.0",
        "status": "healthy",
        "available_analyses": [
            "transcription",
            "objects_tracking",
            "scene_detection",
            "face_tracking",
            "action_recognition",
            "temporal_analysis",
            "audio_analysis",
            "thumbnail_generation",
            "highlight_detection"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": analyzer.models_loaded,
        "memory_usage": analyzer.get_memory_usage(),
        "gpu_available": analyzer.gpu_available,
        "ffmpeg_available": analyzer.ffmpeg_available
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    configuration: str = Form("{}")
):
    \"\"\"Analyze an uploaded video file\"\"\"
    try:
        import json
        config = json.loads(configuration) if configuration else {}
        
        logger.info(f"Starting {analysis_type} analysis for {file.filename}")
        
        # Validate file type
        if not file.content_type.startswith('video/'):
            raise HTTPException(status_code=400, detail="File must be a video")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            result = await analyzer.analyze(
                video_path=tmp_path,
                analysis_type=analysis_type,
                config=config
            )
            
            return AnalysisResponse(**result)
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal analysis error")

@app.post("/batch-analyze")
async def batch_analyze_video(
    file: UploadFile = File(...),
    analysis_types: str = Form(...),
    configuration: str = Form("{}")
):
    \"\"\"Analyze a video with multiple analysis types\"\"\"
    try:
        import json
        analysis_types_list = json.loads(analysis_types)
        config = json.loads(configuration) if configuration else {}
        
        logger.info(f"Starting batch analysis for {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            results = {}
            for analysis_type in analysis_types_list:
                result = await analyzer.analyze(
                    video_path=tmp_path,
                    analysis_type=analysis_type,
                    config=config
                )
                results[analysis_type] = result
            
            return {
                "filename": file.filename,
                "analyses": results,
                "total_analyses": len(analysis_types_list)
            }
            
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")

@app.post("/async-analyze")
async def async_analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    analysis_types: str = Form(...),
    configuration: str = Form("{}"),
    webhook_url: Optional[str] = Form(None)
):
    \"\"\"Start asynchronous video analysis (for large files)\"\"\"
    try:
        import json
        import uuid
        
        analysis_types_list = json.loads(analysis_types)
        config = json.loads(configuration) if configuration else {}
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded file for processing
        upload_dir = "/app/temp"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # Initialize job status
        processing_jobs[job_id] = VideoProcessingJob(
            job_id=job_id,
            status="queued",
            progress=0.0
        )
        
        # Start background processing
        background_tasks.add_task(
            process_video_async,
            job_id,
            file_path,
            analysis_types_list,
            config,
            webhook_url
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Video analysis started. Use /job/{job_id} to check status."
        }
        
    except Exception as e:
        logger.error(f"Async analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start async analysis")

@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    \"\"\"Get status of asynchronous job\"\"\"
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id].dict()

@app.post("/extract-frames")
async def extract_frames(
    file: UploadFile = File(...),
    frame_interval: int = Form(1),
    max_frames: int = Form(100)
):
    \"\"\"Extract frames from video\"\"\"
    try:
        logger.info(f"Extracting frames from {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            frames = await analyzer.extract_frames(
                video_path=tmp_path,
                frame_interval=frame_interval,
                max_frames=max_frames
            )
            
            return {
                "total_frames": len(frames),
                "frame_interval": frame_interval,
                "frames": frames  # Base64 encoded frames
            }
            
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Frame extraction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Frame extraction failed")

@app.post("/generate-thumbnail")
async def generate_thumbnail(
    file: UploadFile = File(...),
    timestamp: float = Form(0.0),
    width: int = Form(320),
    height: int = Form(240)
):
    \"\"\"Generate thumbnail from video\"\"\"
    try:
        logger.info(f"Generating thumbnail for {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            thumbnail = await analyzer.generate_thumbnail(
                video_path=tmp_path,
                timestamp=timestamp,
                width=width,
                height=height
            )
            
            return {
                "thumbnail": thumbnail,  # Base64 encoded
                "timestamp": timestamp,
                "dimensions": {"width": width, "height": height}
            }
            
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"Thumbnail generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")

@app.get("/models")
async def list_models():
    \"\"\"List all loaded models and their information\"\"\"
    return analyzer.get_model_info()

@app.get("/supported-formats")
async def get_supported_formats():
    return {
        "supported_formats": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"],
        "recommended_formats": [".mp4", ".mov"],
        "max_file_size": "500MB",
        "max_duration": "3600s"
    }

async def process_video_async(
    job_id: str,
    file_path: str,
    analysis_types: List[str],
    config: Dict[str, Any],
    webhook_url: Optional[str] = None
):
    \"\"\"Process video asynchronously\"\"\"
    try:
        # Update job status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].progress = 0.0
        
        results = {}
        total_analyses = len(analysis_types)
        
        for i, analysis_type in enumerate(analysis_types):
            try:
                result = await analyzer.analyze(
                    video_path=file_path,
                    analysis_type=analysis_type,
                    config=config
                )
                results[analysis_type] = result
                
                # Update progress
                progress = (i + 1) / total_analyses * 100
                processing_jobs[job_id].progress = progress
                
            except Exception as e:
                logger.error(f"Error in {analysis_type}: {str(e)}")
                results[analysis_type] = {"error": str(e)}
        
        # Mark as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100.0
        processing_jobs[job_id].results = results
        
        # Send webhook if provided
        if webhook_url:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(webhook_url, json={
                        "job_id": job_id,
                        "status": "completed",
                        "results": results
                    })
            except Exception as e:
                logger.error(f"Webhook error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Async processing error: {str(e)}")
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].error = str(e)
    
    finally:
        # Clean up file
        try:
            os.unlink(file_path)
        except:
            pass

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8003)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
