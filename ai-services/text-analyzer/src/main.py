from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
from loguru import logger

from analyzer import TextAnalyzer

app = FastAPI(
    title="Text Analyzer Service",
    description="Advanced NLP analysis service for multimodal content analyzer",
    version="1.0.0"
)

# Initialize the analyzer
analyzer = TextAnalyzer()

class AnalysisRequest(BaseModel):
    text_content: str
    analysis_type: str
    configuration: Optional[Dict[str, Any]] = {}
    language: Optional[str] = "auto"

class AnalysisResponse(BaseModel):
    analysis_type: str
    results: Dict[str, Any]
    confidence: Optional[float] = None
    processing_time: float
    model_version: str
    service_version: str = "1.0.0"

class BatchAnalysisRequest(BaseModel):
    texts: List[str]
    analysis_types: List[str]
    configuration: Optional[Dict[str, Any]] = {}
    language: Optional[str] = "auto"

@app.get("/")
async def root():
    return {
        "service": "Text Analyzer",
        "version": "1.0.0",
        "status": "healthy",
        "available_analyses": [
            "sentiment",
            "entities", 
            "topics",
            "summary",
            "emotions",
            "classification",
            "similarity"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": analyzer.models_loaded,
        "memory_usage": analyzer.get_memory_usage()
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    \"\"\"Analyze a single text with specified analysis type\"\"\"
    try:
        logger.info(f"Starting {request.analysis_type} analysis")
        
        result = await analyzer.analyze(
            text=request.text_content,
            analysis_type=request.analysis_type,
            config=request.configuration,
            language=request.language
        )
        
        return AnalysisResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal analysis error")

@app.post("/batch-analyze")
async def batch_analyze(request: BatchAnalysisRequest):
    \"\"\"Analyze multiple texts with multiple analysis types\"\"\"
    try:
        results = []
        
        for text in request.texts:
            text_results = {}
            for analysis_type in request.analysis_types:
                result = await analyzer.analyze(
                    text=text,
                    analysis_type=analysis_type,
                    config=request.configuration,
                    language=request.language
                )
                text_results[analysis_type] = result
            
            results.append({
                "text": text[:100] + "..." if len(text) > 100 else text,
                "analyses": text_results
            })
        
        return {
            "results": results,
            "total_processed": len(request.texts),
            "analysis_types": request.analysis_types
        }
        
    except Exception as e:
        logger.error(f"Batch analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail="Batch analysis failed")

@app.post("/similarity")
async def compute_similarity(text1: str, text2: str):
    \"\"\"Compute semantic similarity between two texts\"\"\"
    try:
        similarity_score = await analyzer.compute_similarity(text1, text2)
        return {
            "text1": text1[:100] + "..." if len(text1) > 100 else text1,
            "text2": text2[:100] + "..." if len(text2) > 100 else text2,
            "similarity_score": similarity_score,
            "interpretation": analyzer.interpret_similarity(similarity_score)
        }
    except Exception as e:
        logger.error(f"Similarity computation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Similarity computation failed")

@app.get("/models")
async def list_models():
    \"\"\"List all loaded models and their information\"\"\"
    return analyzer.get_model_info()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
