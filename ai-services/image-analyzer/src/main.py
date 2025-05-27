from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import tempfile
import shutil
from loguru import logger

from analyzer import ImageAnalyzer

app = FastAPI(
    title="Image Analyzer Service",
    description="Advanced Computer Vision analysis service for multimodal content analyzer",
    version="1.0.0"
)

# Initialize the analyzer
analyzer = ImageAnalyzer()

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

@app.get("/")
async def root():
    return {
        "service": "Image Analyzer",
        "version": "1.0.0",
        "status": "healthy",
        "available_analyses": [
            "objects",
            "scenes", 
            "ocr",
            "faces",
            "classification",
            "caption",
            "similarity",
            "aesthetic",
            "colors"
        ]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": analyzer.models_loaded,
        "memory_usage": analyzer.get_memory_usage(),
        "gpu_available": analyzer.gpu_available
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    configuration: str = Form("{}")
):
    \"\"\"Analyze an uploaded image\"\"\"
    try:
        import json
        config = json.loads(configuration) if configuration else {}
        
        logger.info(f"Starting {analysis_type} analysis for {file.filename}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            result = await analyzer.analyze(
                image_path=tmp_path,
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
async def batch_analyze_image(
    file: UploadFile = File(...),
    analysis_types: str = Form(...),
    configuration: str = Form("{}")
):
    \"\"\"Analyze an image with multiple analysis types\"\"\"
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
                    image_path=tmp_path,
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

@app.post("/compare-images")
async def compare_images(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    comparison_type: str = Form("similarity")
):
    \"\"\"Compare two images\"\"\"
    try:
        logger.info(f"Comparing images: {image1.filename} vs {image2.filename}")
        
        # Save uploaded files temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image1.filename)[1]) as tmp_file1:
            shutil.copyfileobj(image1.file, tmp_file1)
            tmp_path1 = tmp_file1.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image2.filename)[1]) as tmp_file2:
            shutil.copyfileobj(image2.file, tmp_file2)
            tmp_path2 = tmp_file2.name
        
        try:
            similarity_score = await analyzer.compare_images(tmp_path1, tmp_path2, comparison_type)
            
            return {
                "image1": image1.filename,
                "image2": image2.filename,
                "comparison_type": comparison_type,
                "similarity_score": similarity_score,
                "interpretation": analyzer.interpret_similarity(similarity_score)
            }
            
        finally:
            # Clean up temporary files
            os.unlink(tmp_path1)
            os.unlink(tmp_path2)
        
    except Exception as e:
        logger.error(f"Image comparison error: {str(e)}")
        raise HTTPException(status_code=500, detail="Image comparison failed")

@app.post("/extract-text")
async def extract_text_from_image(
    file: UploadFile = File(...),
    ocr_engine: str = Form("tesseract"),
    language: str = Form("en")
):
    \"\"\"Extract text from image using OCR\"\"\"
    try:
        logger.info(f"Extracting text from {file.filename} using {ocr_engine}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            result = await analyzer.extract_text(tmp_path, ocr_engine, language)
            return result
            
        finally:
            os.unlink(tmp_path)
        
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        raise HTTPException(status_code=500, detail="Text extraction failed")

@app.get("/models")
async def list_models():
    \"\"\"List all loaded models and their information\"\"\"
    return analyzer.get_model_info()

@app.get("/supported-formats")
async def get_supported_formats():
    \"\"\"Get list of supported image formats\"\"\"
    return {
        "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"],
        "recommended_formats": [".jpg", ".png"],
        "max_file_size": "50MB",
        "max_dimensions": "4096x4096"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=os.getenv("ENVIRONMENT") == "development"
    )
