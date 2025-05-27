import asyncio
import time
import psutil
import torch
import cv2
import numpy as np
from PIL import Image, ImageStat
from typing import Dict, List, Any, Optional, Tuple
import torchvision.transforms as transforms
from ultralytics import YOLO
import pytesseract
import easyocr
import face_recognition
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
import os
import tempfile

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    logger.warning("CLIP not available. Some features will be limited.")

class ImageAnalyzer:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        logger.info(f"Using device: {self.device}")
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize EasyOCR
        self.easyocr_reader = None
        
        # Initialize models
        asyncio.create_task(self._load_models())
    
    async def _load_models(self):
        \"\"\"Load all required computer vision models\"\"\"
        try:
            logger.info("Loading computer vision models...")
            
            # Object Detection - YOLO
            logger.info("Loading YOLO model...")
            self.models['yolo'] = YOLO('yolov8n.pt')  # Nano version for faster inference
            
            # Scene Classification - Using a pre-trained ResNet
            logger.info("Loading scene classification model...")
            from torchvision.models import resnet50, ResNet50_Weights
            self.models['scene_classifier'] = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            self.models['scene_classifier'].eval()
            self.models['scene_classifier'].to(self.device)
            
            # Image preprocessing
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            # CLIP for image-text understanding (if available)
            if CLIP_AVAILABLE:
                logger.info("Loading CLIP model...")
                self.models['clip_model'], self.models['clip_preprocess'] = clip.load("ViT-B/32", device=self.device)
            
            # Initialize EasyOCR
            logger.info("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(['en'])
            
            # Load ImageNet class labels
            self._load_imagenet_labels()
            
            self.models_loaded = True
            logger.info("All computer vision models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    def _load_imagenet_labels(self):
        \"\"\"Load ImageNet class labels\"\"\"
        # Simplified version - in production, load from file
        self.imagenet_labels = [
            'tench', 'goldfish', 'great_white_shark', 'tiger_shark', 'hammerhead', 
            'electric_ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
            # ... (truncated for brevity, would include all 1000 ImageNet labels)
        ]
    
    async def analyze(self, image_path: str, analysis_type: str, config: Dict = None) -> Dict[str, Any]:
        \"\"\"Main analysis method\"\"\"
        if not self.models_loaded:
            raise ValueError("Models not loaded yet. Please wait.")
        
        config = config or {}
        start_time = time.time()
        
        try:
            # Load and validate image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError("Invalid image file")
            
            if analysis_type == "objects":
                result = await self._detect_objects(image, config)
            elif analysis_type == "scenes":
                result = await self._classify_scene(image, config)
            elif analysis_type == "ocr":
                result = await self._extract_text_ocr(image_path, config)
            elif analysis_type == "faces":
                result = await self._detect_faces(image, config)
            elif analysis_type == "classification":
                result = await self._classify_image(image, config)
            elif analysis_type == "caption":
                result = await self._generate_caption(image, config)
            elif analysis_type == "aesthetic":
                result = await self._analyze_aesthetics(image, config)
            elif analysis_type == "colors":
                result = await self._analyze_colors(image, config)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            processing_time = time.time() - start_time
            
            return {
                "analysis_type": analysis_type,
                "results": result,
                "confidence": result.get("confidence"),
                "processing_time": processing_time,
                "model_version": self._get_model_version(analysis_type)
            }
            
        except Exception as e:
            logger.error(f"Analysis error for {analysis_type}: {str(e)}")
            raise
    
    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        \"\"\"Load and validate image\"\"\"
        try:
            # Load with OpenCV
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            return None
    
    async def _detect_objects(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Detect objects in image using YOLO\"\"\"
        try:
            confidence_threshold = config.get('confidence_threshold', 0.5)
            
            # Run YOLO detection
            results = self.models['yolo'](image)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        conf = float(box.conf)
                        if conf >= confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            
                            # Get class name
                            class_id = int(box.cls)
                            class_name = self.models['yolo'].names[class_id]
                            
                            detections.append({
                                "class": class_name,
                                "confidence": conf,
                                "bbox": {
                                    "x1": int(x1), "y1": int(y1),
                                    "x2": int(x2), "y2": int(y2)
                                },
                                "area": int((x2 - x1) * (y2 - y1))
                            })
            
            # Sort by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "objects": detections,
                "total_objects": len(detections),
                "unique_classes": list(set([d['class'] for d in detections])),
                "confidence": detections[0]['confidence'] if detections else 0
            }
        except Exception as e:
            logger.error(f"Object detection error: {str(e)}")
            raise
    
    async def _classify_scene(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Classify scene/environment in image\"\"\"
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)
            
            # Preprocess image
            input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.models['scene_classifier'](input_tensor)
                probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            
            # Get top predictions
            top_k = config.get('top_k', 5)
            top_probs, top_indices = torch.topk(probabilities, top_k)
            
            predictions = []
            for i in range(top_k):
                idx = top_indices[i].item()
                prob = top_probs[i].item()
                
                # Use ImageNet labels (simplified)
                label = self.imagenet_labels[idx] if idx < len(self.imagenet_labels) else f"class_{idx}"
                
                predictions.append({
                    "label": label,
                    "confidence": prob
                })
            
            return {
                "scene_predictions": predictions,
                "top_scene": predictions[0]["label"] if predictions else "unknown",
                "confidence": predictions[0]["confidence"] if predictions else 0
            }
        except Exception as e:
            logger.error(f"Scene classification error: {str(e)}")
            raise
    
    async def _extract_text_ocr(self, image_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Extract text using OCR\"\"\"
        try:
            ocr_engine = config.get('ocr_engine', 'tesseract')
            language = config.get('language', 'en')
            
            if ocr_engine == 'tesseract':
                # Use Tesseract
                text = pytesseract.image_to_string(Image.open(image_path), lang=language)
                # Get detailed data
                data = pytesseract.image_to_data(Image.open(image_path), output_type=pytesseract.Output.DICT)
                
                words = []
                for i in range(len(data['text'])):
                    if int(data['conf'][i]) > 0:
                        words.append({
                            "text": data['text'][i],
                            "confidence": int(data['conf'][i]) / 100.0,
                            "bbox": {
                                "x": data['left'][i],
                                "y": data['top'][i],
                                "width": data['width'][i],
                                "height": data['height'][i]
                            }
                        })
                
            elif ocr_engine == 'easyocr':
                # Use EasyOCR
                results = self.easyocr_reader.readtext(image_path)
                text = " ".join([result[1] for result in results])
                
                words = []
                for result in results:
                    bbox, extracted_text, confidence = result
                    words.append({
                        "text": extracted_text,
                        "confidence": confidence,
                        "bbox": {
                            "points": bbox
                        }
                    })
            
            else:
                raise ValueError(f"Unsupported OCR engine: {ocr_engine}")
            
            return {
                "extracted_text": text.strip(),
                "words": words,
                "total_words": len([w for w in words if w['text'].strip()]),
                "ocr_engine": ocr_engine,
                "average_confidence": np.mean([w['confidence'] for w in words]) if words else 0
            }
        except Exception as e:
            logger.error(f"OCR error: {str(e)}")
            raise
    
    async def _detect_faces(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Detect faces in image\"\"\"
        try:
            method = config.get('method', 'face_recognition')
            
            if method == 'face_recognition':
                # Use face_recognition library
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)
                
                faces = []
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    faces.append({
                        "face_id": i,
                        "bbox": {
                            "x1": left, "y1": top,
                            "x2": right, "y2": bottom
                        },
                        "area": (right - left) * (bottom - top),
                        "encoding": face_encodings[i].tolist() if i < len(face_encodings) else None
                    })
            
            elif method == 'mediapipe':
                # Use MediaPipe
                with self.mp_face_detection.FaceDetection(
                    model_selection=0, min_detection_confidence=0.5
                ) as face_detection:
                    
                    results = face_detection.process(image)
                    faces = []
                    
                    if results.detections:
                        for i, detection in enumerate(results.detections):
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = image.shape
                            
                            faces.append({
                                "face_id": i,
                                "bbox": {
                                    "x1": int(bbox.xmin * w),
                                    "y1": int(bbox.ymin * h),
                                    "x2": int((bbox.xmin + bbox.width) * w),
                                    "y2": int((bbox.ymin + bbox.height) * h)
                                },
                                "confidence": detection.score[0],
                                "keypoints": self._extract_face_keypoints(detection)
                            })
            
            return {
                "faces": faces,
                "total_faces": len(faces),
                "method": method,
                "largest_face": max(faces, key=lambda x: x.get('area', 0)) if faces else None
            }
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            raise
    
    def _extract_face_keypoints(self, detection) -> Dict[str, Any]:
        \"\"\"Extract face keypoints from MediaPipe detection\"\"\"
        # Simplified keypoint extraction
        return {
            "nose_tip": "detected",
            "eyes": "detected",
            "mouth": "detected"
        }
    
    async def _classify_image(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"General image classification\"\"\"
        try:
            # Use the same scene classifier for general classification
            return await self._classify_scene(image, config)
        except Exception as e:
            logger.error(f"Image classification error: {str(e)}")
            raise
    
    async def _generate_caption(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Generate image caption (simplified version)\"\"\"
        try:
            # This is a simplified version
            # In a real implementation, you'd use a proper image captioning model
            
            # For now, combine object detection and scene classification
            objects_result = await self._detect_objects(image, {"confidence_threshold": 0.3})
            scene_result = await self._classify_scene(image, {"top_k": 1})
            
            objects = [obj["class"] for obj in objects_result["objects"][:3]]
            scene = scene_result["top_scene"] if scene_result["scene_predictions"] else "scene"
            
            if objects:
                caption = f"An image showing {', '.join(objects)} in a {scene}"
            else:
                caption = f"An image of a {scene}"
            
            return {
                "caption": caption,
                "confidence": 0.7,  # Mock confidence
                "method": "rule_based",
                "detected_objects": objects,
                "scene": scene
            }
        except Exception as e:
            logger.error(f"Caption generation error: {str(e)}")
            raise
    
    async def _analyze_aesthetics(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze image aesthetics\"\"\"
        try:
            pil_image = Image.fromarray(image)
            
            # Basic aesthetic metrics
            
            # 1. Color analysis
            stat = ImageStat.Stat(pil_image)
            
            # 2. Brightness
            brightness = sum(stat.mean) / 3
            
            # 3. Contrast (simplified)
            contrast = max(stat.stddev) if stat.stddev else 0
            
            # 4. Saturation (for RGB images)
            if len(stat.mean) >= 3:
                r, g, b = stat.mean[:3]
                max_rgb = max(r, g, b)
                min_rgb = min(r, g, b)
                saturation = (max_rgb - min_rgb) / max_rgb if max_rgb > 0 else 0
            else:
                saturation = 0
            
            # 5. Rule of thirds (simplified)
            h, w = image.shape[:2]
            rule_of_thirds_score = self._analyze_rule_of_thirds(image)
            
            # Overall aesthetic score (simplified heuristic)
            aesthetic_score = (
                min(brightness / 128, 1) * 0.3 +
                min(contrast / 50, 1) * 0.3 +
                saturation * 0.2 +
                rule_of_thirds_score * 0.2
            )
            
            return {
                "aesthetic_score": aesthetic_score,
                "brightness": brightness / 255,
                "contrast": min(contrast / 100, 1),
                "saturation": saturation,
                "rule_of_thirds_score": rule_of_thirds_score,
                "interpretation": self._interpret_aesthetics(aesthetic_score)
            }
        except Exception as e:
            logger.error(f"Aesthetic analysis error: {str(e)}")
            raise
    
    def _analyze_rule_of_thirds(self, image: np.ndarray) -> float:
        \"\"\"Analyze rule of thirds composition (simplified)\"\"\"
        # This is a very simplified version
        # Real implementation would use edge detection and interest point analysis
        return 0.5  # Mock score
    
    def _interpret_aesthetics(self, score: float) -> str:
        \"\"\"Interpret aesthetic score\"\"\"
        if score > 0.8:
            return "Highly aesthetic"
        elif score > 0.6:
            return "Good aesthetics"
        elif score > 0.4:
            return "Average aesthetics"
        else:
            return "Poor aesthetics"
    
    async def _analyze_colors(self, image: np.ndarray, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze color composition\"\"\"
        try:
            pil_image = Image.fromarray(image)
            
            # Convert to different color spaces for analysis
            hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Dominant colors using k-means clustering
            from sklearn.cluster import KMeans
            
            # Reshape image to be a list of pixels
            pixels = image.reshape(-1, 3)
            
            # Use k-means to find dominant colors
            n_colors = config.get('n_colors', 5)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get dominant colors and their percentages
            colors = kmeans.cluster_centers_.astype(int)
            labels = kmeans.labels_
            
            color_percentages = []
            for i, color in enumerate(colors):
                percentage = (labels == i).sum() / len(labels)
                color_percentages.append({
                    "rgb": color.tolist(),
                    "hex": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}",
                    "percentage": percentage,
                    "color_name": self._get_color_name(color)
                })
            
            # Sort by percentage
            color_percentages.sort(key=lambda x: x['percentage'], reverse=True)
            
            # Color temperature analysis
            avg_color = np.mean(pixels, axis=0)
            color_temperature = self._analyze_color_temperature(avg_color)
            
            return {
                "dominant_colors": color_percentages,
                "color_temperature": color_temperature,
                "average_color": {
                    "rgb": avg_color.astype(int).tolist(),
                    "hex": f"#{int(avg_color[0]):02x}{int(avg_color[1]):02x}{int(avg_color[2]):02x}"
                },
                "color_harmony": self._analyze_color_harmony(color_percentages)
            }
        except Exception as e:
            logger.error(f"Color analysis error: {str(e)}")
            raise
    
    def _get_color_name(self, rgb: np.ndarray) -> str:
        \"\"\"Get approximate color name\"\"\"
        r, g, b = rgb
        
        # Simplified color naming
        if r > 200 and g > 200 and b > 200:
            return "White"
        elif r < 50 and g < 50 and b < 50:
            return "Black"
        elif r > g and r > b:
            return "Red"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        elif r > 150 and g > 150 and b < 100:
            return "Yellow"
        elif r > 150 and g < 100 and b > 150:
            return "Magenta"
        elif r < 100 and g > 150 and b > 150:
            return "Cyan"
        else:
            return "Mixed"
    
    def _analyze_color_temperature(self, avg_color: np.ndarray) -> str:
        \"\"\"Analyze color temperature\"\"\"
        r, g, b = avg_color
        
        if r > b + 30:
            return "Warm"
        elif b > r + 30:
            return "Cool"
        else:
            return "Neutral"
    
    def _analyze_color_harmony(self, colors: List[Dict]) -> str:
        \"\"\"Analyze color harmony (simplified)\"\"\"
        if len(colors) < 2:
            return "Monochromatic"
        
        # Simplified harmony analysis
        dominant_colors = colors[:3]
        
        # Check if colors are similar (analogous)
        # This is a very simplified version
        return "Complementary"  # Mock result
    
    async def compare_images(self, image_path1: str, image_path2: str, comparison_type: str) -> float:
        \"\"\"Compare two images\"\"\"
        try:
            image1 = self._load_image(image_path1)
            image2 = self._load_image(image_path2)
            
            if image1 is None or image2 is None:
                raise ValueError("Could not load one or both images")
            
            if comparison_type == "similarity":
                return self._compute_image_similarity(image1, image2)
            elif comparison_type == "structural":
                return self._compute_structural_similarity(image1, image2)
            else:
                raise ValueError(f"Unsupported comparison type: {comparison_type}")
        except Exception as e:
            logger.error(f"Image comparison error: {str(e)}")
            raise
    
    def _compute_image_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        \"\"\"Compute basic image similarity using histograms\"\"\"
        # Resize images to same size
        h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
        image1_resized = cv2.resize(image1, (w, h))
        image2_resized = cv2.resize(image2, (w, h))
        
        # Calculate histograms
        hist1 = cv2.calcHist([image1_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([image2_resized], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        # Calculate correlation
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return float(similarity)
    
    def _compute_structural_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        \"\"\"Compute structural similarity\"\"\"
        from skimage.metrics import structural_similarity as ssim
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
        
        # Resize to same dimensions
        h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
        gray1_resized = cv2.resize(gray1, (w, h))
        gray2_resized = cv2.resize(gray2, (w, h))
        
        # Compute SSIM
        similarity = ssim(gray1_resized, gray2_resized)
        return float(similarity)
    
    async def extract_text(self, image_path: str, ocr_engine: str, language: str) -> Dict[str, Any]:
        \"\"\"Extract text from image\"\"\"
        return await self._extract_text_ocr(image_path, {
            'ocr_engine': ocr_engine,
            'language': language
        })
    
    def interpret_similarity(self, score: float) -> str:
        \"\"\"Interpret similarity score\"\"\"
        if score > 0.9:
            return "Nearly identical"
        elif score > 0.7:
            return "Very similar"
        elif score > 0.5:
            return "Similar"
        elif score > 0.3:
            return "Somewhat similar"
        else:
            return "Different"
    
    def _get_model_version(self, analysis_type: str) -> str:
        \"\"\"Get model version for specific analysis type\"\"\"
        model_versions = {
            "objects": "YOLOv8n",
            "scenes": "ResNet50-ImageNet",
            "faces": "face_recognition",
            "classification": "ResNet50-ImageNet",
            "ocr": "Tesseract-5.0"
        }
        return model_versions.get(analysis_type, "unknown")
    
    def get_memory_usage(self) -> Dict[str, float]:
        \"\"\"Get current memory usage\"\"\"
        process = psutil.Process()
        memory_info = process.memory_info()
        
        gpu_memory = {}
        if self.gpu_available:
            try:
                gpu_memory = {
                    "gpu_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                    "gpu_cached_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                }
            except:
                pass
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            **gpu_memory
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        \"\"\"Get information about loaded models\"\"\"
        return {
            "models_loaded": self.models_loaded,
            "device": self.device,
            "gpu_available": self.gpu_available,
            "available_analyses": [
                "objects", "scenes", "ocr", "faces", 
                "classification", "caption", "aesthetic", "colors"
            ],
            "loaded_models": list(self.models.keys()),
            "memory_usage": self.get_memory_usage()
        }
