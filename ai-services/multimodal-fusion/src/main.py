from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import uvicorn
import os
import asyncio
from loguru import logger

from fusion_engine import MultimodalFusionEngine

app = FastAPI(
    title="Multimodal Fusion Service",
    description="Advanced multimodal analysis fusion service that combines insights from text, image, and video analysis",
    version="1.0.0"
)

# Initialize the fusion engine
fusion_engine = MultimodalFusionEngine()

class FusionRequest(BaseModel):
    analysis_results: Dict[str, Any]  # Results from different modalities
    fusion_types: List[str] = ["semantic", "temporal", "cross_modal"]
    configuration: Optional[Dict[str, Any]] = {}

class CrossModalRequest(BaseModel):
    text_results: Optional[Dict[str, Any]] = None
    image_results: Optional[Dict[str, Any]] = None  
    video_results: Optional[Dict[str, Any]] = None
    correlation_types: List[str] = ["semantic", "entity", "sentiment", "temporal"]
    configuration: Optional[Dict[str, Any]] = {}

class InsightGenerationRequest(BaseModel):
    multimodal_results: Dict[str, Any]
    insight_types: List[str] = ["patterns", "anomalies", "trends", "relationships"]
    configuration: Optional[Dict[str, Any]] = {}

class FusionResponse(BaseModel):
    fusion_type: str
    results: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: float
    model_version: str = "multimodal_fusion_v1.0"
    service_version: str = "1.0.0"

class MultimodalJob(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# In-memory job storage
fusion_jobs = {}

@app.get("/")
async def root():
    return {
        "service": "Multimodal Fusion",
        "version": "1.0.0",
        "status": "healthy",
        "capabilities": [
            "semantic_fusion",
            "temporal_alignment", 
            "cross_modal_correlation",
            "insight_generation",
            "pattern_discovery",
            "anomaly_detection",
            "relationship_mapping",
            "narrative_synthesis"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": fusion_engine.models_loaded,
        "memory_usage": fusion_engine.get_memory_usage(),
        "fusion_algorithms": fusion_engine.get_available_algorithms()
    }

@app.post("/fuse", response_model=FusionResponse)
async def fuse_multimodal_data(request: FusionRequest):
    \"\"\"Fuse analysis results from multiple modalities\"\"\"
    try:
        logger.info(f"Starting multimodal fusion with types: {request.fusion_types}")
        
        result = await fusion_engine.fuse_modalities(
            analysis_results=request.analysis_results,
            fusion_types=request.fusion_types,
            config=request.configuration
        )
        
        return FusionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Fusion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Multimodal fusion failed")

@app.post("/correlate")
async def cross_modal_correlation(request: CrossModalRequest):
    \"\"\"Find correlations between different modalities\"\"\"
    try:
        logger.info(f"Starting cross-modal correlation with types: {request.correlation_types}")
        
        results = await fusion_engine.correlate_modalities(
            text_results=request.text_results,
            image_results=request.image_results,
            video_results=request.video_results,
            correlation_types=request.correlation_types,
            config=request.configuration
        )
        
        return {
            "correlations": results,
            "total_correlations": len(results),
            "correlation_types": request.correlation_types
        }
        
    except Exception as e:
        logger.error(f"Correlation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Cross-modal correlation failed")

@app.post("/generate-insights")
async def generate_insights(request: InsightGenerationRequest):
    \"\"\"Generate high-level insights from multimodal analysis\"\"\"
    try:
        logger.info(f"Generating insights of types: {request.insight_types}")
        
        insights = await fusion_engine.generate_insights(
            multimodal_results=request.multimodal_results,
            insight_types=request.insight_types,  
            config=request.configuration
        )
        
        return {
            "insights": insights,
            "insight_types": request.insight_types,
            "confidence": insights.get("overall_confidence", 0.7)
        }
        
    except Exception as e:
        logger.error(f"Insight generation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Insight generation failed")

@app.post("/semantic-alignment")
async def semantic_alignment(
    text_content: str,
    image_descriptions: List[str],
    video_transcripts: List[str]
):
    \"\"\"Align content semantically across modalities\"\"\"
    try:
        logger.info("Starting semantic alignment")
        
        alignment = await fusion_engine.align_semantically(
            text_content=text_content,
            image_descriptions=image_descriptions,
            video_transcripts=video_transcripts
        )
        
        return {
            "semantic_alignment": alignment,
            "coherence_score": alignment.get("coherence_score", 0.0),
            "alignment_matrix": alignment.get("alignment_matrix", [])
        }
        
    except Exception as e:
        logger.error(f"Semantic alignment error: {str(e)}")
        raise HTTPException(status_code=500, detail="Semantic alignment failed")

@app.post("/temporal-synchronization")
async def temporal_synchronization(
    video_timeline: List[Dict[str, Any]],
    audio_events: List[Dict[str, Any]],
    text_timestamps: List[Dict[str, Any]]
):
    \"\"\"Synchronize events across temporal modalities\"\"\"
    try:
        logger.info("Starting temporal synchronization")
        
        sync_result = await fusion_engine.synchronize_temporal(
            video_timeline=video_timeline,
            audio_events=audio_events,
            text_timestamps=text_timestamps
        )
        
        return {
            "synchronized_timeline": sync_result["timeline"],
            "synchronization_quality": sync_result.get("quality_score", 0.0),
            "temporal_conflicts": sync_result.get("conflicts", [])
        }
        
    except Exception as e:
        logger.error(f"Temporal synchronization error: {str(e)}")
        raise HTTPException(status_code=500, detail="Temporal synchronization failed")

@app.post("/pattern-discovery")
async def discover_patterns(
    multimodal_data: Dict[str, Any],
    pattern_types: List[str] = ["recurring", "sequential", "hierarchical"]
):
    \"\"\"Discover patterns across multimodal data\"\"\"
    try:
        logger.info(f"Discovering patterns of types: {pattern_types}")
        
        patterns = await fusion_engine.discover_patterns(
            multimodal_data=multimodal_data,
            pattern_types=pattern_types
        )
        
        return {
            "discovered_patterns": patterns,
            "pattern_confidence": patterns.get("confidence_scores", {}),
            "pattern_significance": patterns.get("significance_scores", {})
        }
        
    except Exception as e:
        logger.error(f"Pattern discovery error: {str(e)}")
        raise HTTPException(status_code=500, detail="Pattern discovery failed")

@app.post("/anomaly-detection")
async def detect_anomalies(
    multimodal_data: Dict[str, Any],
    anomaly_types: List[str] = ["statistical", "semantic", "temporal"]
):
    \"\"\"Detect anomalies across multimodal data\"\"\"
    try:
        logger.info(f"Detecting anomalies of types: {anomaly_types}")
        
        anomalies = await fusion_engine.detect_anomalies(
            multimodal_data=multimodal_data,
            anomaly_types=anomaly_types
        )
        
        return {
            "detected_anomalies": anomalies,
            "anomaly_scores": anomalies.get("scores", {}),
            "severity_levels": anomalies.get("severity", {})
        }
        
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail="Anomaly detection failed")

@app.post("/narrative-synthesis")
async def synthesize_narrative(
    multimodal_insights: Dict[str, Any],
    narrative_style: str = "comprehensive",
    target_audience: str = "general"
):
    \"\"\"Synthesize a narrative from multimodal insights\"\"\"
    try:
        logger.info(f"Synthesizing narrative in {narrative_style} style for {target_audience} audience")
        
        narrative = await fusion_engine.synthesize_narrative(
            insights=multimodal_insights,
            style=narrative_style,
            audience=target_audience
        )
        
        return {
            "narrative": narrative["text"],
            "key_points": narrative.get("key_points", []),
            "confidence": narrative.get("confidence", 0.7),
            "readability_score": narrative.get("readability", 0.8)
        }
        
    except Exception as e:
        logger.error(f"Narrative synthesis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Narrative synthesis failed")

@app.post("/async-fusion")
async def async_multimodal_fusion(
    background_tasks: BackgroundTasks,
    request: FusionRequest,
    webhook_url: Optional[str] = None
):
    \"\"\"Start asynchronous multimodal fusion for complex analysis\"\"\"
    try:
        import uuid
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        fusion_jobs[job_id] = MultimodalJob(
            job_id=job_id,
            status="queued",
            progress=0.0
        )
        
        # Start background processing
        background_tasks.add_task(
            process_fusion_async,
            job_id,
            request,
            webhook_url
        )
        
        return {
            "job_id": job_id,
            "status": "queued",
            "message": "Multimodal fusion started. Use /job/{job_id} to check status."
        }
        
    except Exception as e:
        logger.error(f"Async fusion error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to start async fusion")

@app.get("/job/{job_id}")
async def get_fusion_job_status(job_id: str):
    \"\"\"Get status of asynchronous fusion job\"\"\"
    if job_id not in fusion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return fusion_jobs[job_id].dict()

@app.get("/algorithms")
async def list_algorithms():
    \"\"\"List available fusion algorithms\"\"\"
    return fusion_engine.get_algorithm_details()

@app.get("/models")
async def list_models():
    \"\"\"List all loaded models and their information\"\"\"
    return fusion_engine.get_model_info()

async def process_fusion_async(
    job_id: str,
    request: FusionRequest,
    webhook_url: Optional[str] = None
):
    \"\"\"Process multimodal fusion asynchronously\"\"\"
    try:
        # Update job status
        fusion_jobs[job_id].status = "processing"
        fusion_jobs[job_id].progress = 10.0
        
        # Perform fusion
        result = await fusion_engine.fuse_modalities(
            analysis_results=request.analysis_results,
            fusion_types=request.fusion_types,
            config=request.configuration
        )
        
        fusion_jobs[job_id].progress = 70.0
        
        # Generate additional insights
        insights = await fusion_engine.generate_insights(
            multimodal_results=result["results"],
            insight_types=["patterns", "relationships"],
            config=request.configuration
        )
        
        fusion_jobs[job_id].progress = 90.0
        
        # Combine results
        final_results = {
            "fusion": result,
            "insights": insights,
            "processing_complete": True
        }
        
        # Mark as completed
        fusion_jobs[job_id].status = "completed"
        fusion_jobs[job_id].progress = 100.0
        fusion_jobs[job_id].results = final_results
        
        # Send webhook if provided
        if webhook_url:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    await client.post(webhook_url, json={
                        "job_id": job_id,
                        "status": "completed",
                        "results": final_results
                    })
            except Exception as e:
                logger.error(f"Webhook error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Async fusion processing error: {str(e)}")
        fusion_jobs[job_id].status = "failed"
        fusion_jobs[job_id].error = str(e)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8004)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
