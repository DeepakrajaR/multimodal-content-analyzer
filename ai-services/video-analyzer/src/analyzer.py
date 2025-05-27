import asyncio
import time
import psutil
import torch
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import tempfile
import os
import base64
from PIL import Image
import subprocess
import json
from loguru import logger

# Video processing
import moviepy.editor as mp
from moviepy.video.io.VideoFileClip import VideoFileClip
import ffmpeg

# Audio processing
import whisper
import librosa
import soundfile as sf

# Computer vision for frame analysis
from ultralytics import YOLO
import mediapipe as MediaPipe

# Scene detection
try:
    from scenedetect import VideoManager, SceneManager
    from scenedetect.detectors import ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logger.warning("PySceneDetect not available. Scene detection will be limited.")

class VideoAnalyzer:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gpu_available = torch.cuda.is_available()
        self.ffmpeg_available = self._check_ffmpeg()
        logger.info(f"Using device: {self.device}")
        
        # Initialize MediaPipe
        self.mp_pose = MediaPipe.solutions.pose
        self.mp_hands = MediaPipe.solutions.hands
        self.mp_face_detection = MediaPipe.solutions.face_detection
        
        # Initialize models
        asyncio.create_task(self._load_models())
    
    def _check_ffmpeg(self) -> bool:
        \"\"\"Check if FFmpeg is available\"\"\"
        try:
            subprocess.run(['ffmpeg', '-version'], 
                          capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not found. Some video processing features will be limited.")
            return False
    
    async def _load_models(self):
        \"\"\"Load all required models for video analysis\"\"\"
        try:
            logger.info("Loading video analysis models...")
            
            # Whisper for audio transcription
            logger.info("Loading Whisper model...")
            self.models['whisper'] = whisper.load_model("base")
            
            # YOLO for object detection in frames
            logger.info("Loading YOLO model...")
            self.models['yolo'] = YOLO('yolov8n.pt')
            
            # Face tracking
            self.models['face_detection'] = self.mp_face_detection.FaceDetection(
                model_selection=0, min_detection_confidence=0.5
            )
            
            # Pose estimation
            self.models['pose_estimation'] = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            self.models_loaded = True
            logger.info("All video analysis models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    async def analyze(self, video_path: str, analysis_type: str, config: Dict = None) -> Dict[str, Any]:
        \"\"\"Main analysis method\"\"\"
        if not self.models_loaded:
            raise ValueError("Models not loaded yet. Please wait.")
        
        config = config or {}
        start_time = time.time()
        
        try:
            # Validate video file
            if not os.path.exists(video_path):
                raise ValueError("Video file not found")
            
            # Get video info
            video_info = self._get_video_info(video_path)
            
            if analysis_type == "transcription":
                result = await self._transcribe_audio(video_path, config)
            elif analysis_type == "objects_tracking":
                result = await self._track_objects(video_path, config)
            elif analysis_type == "scene_detection":
                result = await self._detect_scenes(video_path, config)
            elif analysis_type == "face_tracking":
                result = await self._track_faces(video_path, config)
            elif analysis_type == "action_recognition":
                result = await self._recognize_actions(video_path, config)
            elif analysis_type == "temporal_analysis":
                result = await self._analyze_temporal(video_path, config)
            elif analysis_type == "audio_analysis":
                result = await self._analyze_audio(video_path, config)
            elif analysis_type == "thumbnail_generation":
                result = await self._generate_thumbnails(video_path, config)
            elif analysis_type == "highlight_detection":
                result = await self._detect_highlights(video_path, config)
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
            
            processing_time = time.time() - start_time
            
            # Add video metadata to result
            result.update({
                "video_info": video_info,
                "processing_time": processing_time
            })
            
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
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        \"\"\"Get basic video information\"\"\"
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                "duration": duration,
                "fps": fps,
                "frame_count": frame_count,
                "resolution": {"width": width, "height": height},
                "aspect_ratio": width / height if height > 0 else 0,
                "file_size": os.path.getsize(video_path)
            }
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return {}
    
    async def _transcribe_audio(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Transcribe audio from video using Whisper\"\"\"
        try:
            logger.info("Starting audio transcription...")
            
            # Extract audio from video
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            try:
                # Use moviepy to extract audio
                video = VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                
                # Transcribe with Whisper
                result = self.models['whisper'].transcribe(
                    audio_path,
                    language=config.get('language', None),
                    task=config.get('task', 'transcribe')  # transcribe or translate
                )
                
                # Process segments for timeline
                segments = []
                for segment in result.get('segments', []):
                    segments.append({
                        "start": segment['start'],
                        "end": segment['end'],
                        "text": segment['text'].strip(),
                        "confidence": segment.get('avg_logprob', 0)
                    })
                
                return {
                    "transcript": result['text'],
                    "language": result.get('language', 'unknown'),
                    "segments": segments,
                    "total_segments": len(segments),
                    "confidence": np.mean([s['confidence'] for s in segments]) if segments else 0
                }
                
            finally:
                # Clean up temp audio file
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise
    
    async def _track_objects(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Track objects throughout the video\"\"\"
        try:
            logger.info("Starting object tracking...")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_interval = config.get('frame_interval', 30)  # Process every 30th frame
            confidence_threshold = config.get('confidence_threshold', 0.5)
            
            detections_timeline = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process only at specified intervals
                if frame_number % frame_interval == 0:
                    timestamp = frame_number / fps
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Run YOLO detection
                    results = self.models['yolo'](frame_rgb)
                    
                    frame_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                conf = float(box.conf)
                                if conf >= confidence_threshold:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    class_id = int(box.cls)
                                    class_name = self.models['yolo'].names[class_id]
                                    
                                    frame_detections.append({
                                        "class": class_name,
                                        "confidence": conf,
                                        "bbox": {
                                            "x1": int(x1), "y1": int(y1),
                                            "x2": int(x2), "y2": int(y2)
                                        }
                                    })
                    
                    detections_timeline.append({
                        "timestamp": timestamp,
                        "frame_number": frame_number,
                        "detections": frame_detections
                    })
                
                frame_number += 1
            
            cap.release()
            
            # Analyze object statistics
            all_objects = {}
            for frame_data in detections_timeline:
                for detection in frame_data['detections']:
                    obj_class = detection['class']
                    if obj_class not in all_objects:
                        all_objects[obj_class] = {
                            "count": 0,
                            "avg_confidence": 0,
                            "first_seen": frame_data['timestamp'],
                            "last_seen": frame_data['timestamp']
                        }
                    
                    all_objects[obj_class]["count"] += 1
                    all_objects[obj_class]["last_seen"] = frame_data['timestamp']
            
            # Calculate average confidences
            for obj_class in all_objects:
                confidences = []
                for frame_data in detections_timeline:
                    for detection in frame_data['detections']:
                        if detection['class'] == obj_class:
                            confidences.append(detection['confidence'])
                
                if confidences:
                    all_objects[obj_class]["avg_confidence"] = np.mean(confidences)
            
            return {
                "timeline": detections_timeline,
                "object_statistics": all_objects,
                "total_frames_processed": len(detections_timeline),
                "unique_objects": list(all_objects.keys()),
                "processing_interval": frame_interval
            }
            
        except Exception as e:
            logger.error(f"Object tracking error: {str(e)}")
            raise
    
    async def _detect_scenes(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Detect scene changes in video\"\"\"
        try:
            logger.info("Starting scene detection...")
            
            if not SCENEDETECT_AVAILABLE:
                # Fallback to simple frame difference method
                return await self._detect_scenes_simple(video_path, config)
            
            # Use PySceneDetect
            video_manager = VideoManager([video_path])
            scene_manager = SceneManager()
            
            # Add ContentDetector algorithm
            threshold = config.get('threshold', 30.0)
            scene_manager.add_detector(ContentDetector(threshold=threshold))
            
            # Start video manager
            video_manager.set_duration(start_time=0)
            video_manager.start()
            
            # Perform scene detection
            scene_manager.detect_scenes(frame_source=video_manager)
            
            # Get scene list
            scene_list = scene_manager.get_scene_list()
            
            scenes = []
            for i, (start_time, end_time) in enumerate(scene_list):
                scenes.append({
                    "scene_id": i,
                    "start_time": start_time.get_seconds(),
                    "end_time": end_time.get_seconds(),
                    "duration": (end_time - start_time).get_seconds()
                })
            
            video_manager.release()
            
            return {
                "scenes": scenes,
                "total_scenes": len(scenes),
                "detection_threshold": threshold,
                "average_scene_duration": np.mean([s['duration'] for s in scenes]) if scenes else 0
            }
            
        except Exception as e:
            logger.error(f"Scene detection error: {str(e)}")
            raise
    
    async def _detect_scenes_simple(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Simple scene detection using frame differences\"\"\"
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            prev_frame = None
            scenes = []
            current_scene_start = 0
            frame_number = 0
            threshold = config.get('threshold', 0.3)
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(prev_frame, gray)
                    diff_score = np.mean(diff) / 255.0
                    
                    # Detect scene change
                    if diff_score > threshold:
                        timestamp = frame_number / fps
                        scenes.append({
                            "scene_id": len(scenes),
                            "start_time": current_scene_start,
                            "end_time": timestamp,
                            "duration": timestamp - current_scene_start
                        })
                        current_scene_start = timestamp
                
                prev_frame = gray
                frame_number += 1
            
            # Add final scene
            final_timestamp = frame_number / fps
            if current_scene_start < final_timestamp:
                scenes.append({
                    "scene_id": len(scenes),
                    "start_time": current_scene_start,
                    "end_time": final_timestamp,
                    "duration": final_timestamp - current_scene_start
                })
            
            cap.release()
            
            return {
                "scenes": scenes,
                "total_scenes": len(scenes),
                "detection_method": "frame_difference",
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Simple scene detection error: {str(e)}")
            raise
    
    async def _track_faces(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Track faces throughout the video\"\"\"
        try:
            logger.info("Starting face tracking...")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_interval = config.get('frame_interval', 15)  # Process every 15th frame
            
            face_timeline = []
            frame_number = 0
            
            with self.models['face_detection'] as face_detection:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_number % frame_interval == 0:
                        timestamp = frame_number / fps
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Detect faces
                        results = face_detection.process(frame_rgb)
                        
                        frame_faces = []
                        if results.detections:
                            for detection in results.detections:
                                bbox = detection.location_data.relative_bounding_box
                                h, w, _ = frame.shape
                                
                                frame_faces.append({
                                    "bbox": {
                                        "x1": int(bbox.xmin * w),
                                        "y1": int(bbox.ymin * h),
                                        "x2": int((bbox.xmin + bbox.width) * w),
                                        "y2": int((bbox.ymin + bbox.height) * h)
                                    },
                                    "confidence": detection.score[0]
                                })
                        
                        face_timeline.append({
                            "timestamp": timestamp,
                            "frame_number": frame_number,
                            "faces": frame_faces,
                            "face_count": len(frame_faces)
                        })
                    
                    frame_number += 1
            
            cap.release()
            
            # Analyze face statistics
            face_counts = [frame_data['face_count'] for frame_data in face_timeline]
            
            return {
                "timeline": face_timeline,
                "statistics": {
                    "max_faces": max(face_counts) if face_counts else 0,
                    "avg_faces": np.mean(face_counts) if face_counts else 0,
                    "total_frames_with_faces": sum(1 for count in face_counts if count > 0),
                    "face_presence_ratio": (sum(1 for count in face_counts if count > 0) / len(face_counts)) if face_counts else 0
                },
                "total_frames_processed": len(face_timeline)
            }
            
        except Exception as e:
            logger.error(f"Face tracking error: {str(e)}")
            raise
    
    async def _recognize_actions(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Recognize actions/activities in video (simplified version)\"\"\"
        try:
            logger.info("Starting action recognition...")
            
            # This is a simplified version using pose estimation
            # In a full implementation, you'd use action recognition models
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frame_interval = config.get('frame_interval', 30)
            
            pose_timeline = []
            frame_number = 0
            
            with self.models['pose_estimation'] as pose:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_number % frame_interval == 0:
                        timestamp = frame_number / fps
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Process pose
                        results = pose.process(frame_rgb)
                        
                        if results.pose_landmarks:
                            # Extract key pose information
                            landmarks = results.pose_landmarks.landmark
                            
                            # Simplified action detection based on pose
                            action = self._classify_pose_action(landmarks)
                            
                            pose_timeline.append({
                                "timestamp": timestamp,
                                "frame_number": frame_number,
                                "action": action,
                                "confidence": 0.7  # Mock confidence
                            })
                    
                    frame_number += 1
            
            cap.release()
            
            # Analyze action patterns
            actions = [frame_data['action'] for frame_data in pose_timeline]
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            return {
                "timeline": pose_timeline,
                "action_statistics": action_counts,
                "dominant_action": max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else "unknown",
                "total_frames_processed": len(pose_timeline)
            }
            
        except Exception as e:
            logger.error(f"Action recognition error: {str(e)}")
            raise
    
    def _classify_pose_action(self, landmarks) -> str:
        \"\"\"Classify action based on pose landmarks (simplified)\"\"\"
        # This is a very simplified action classification
        # Real implementation would use trained action recognition models
        
        try:
            # Get key points
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            left_elbow = landmarks[13]
            right_elbow = landmarks[14]
            left_wrist = landmarks[15]
            right_wrist = landmarks[16]
            
            # Simple heuristics for action classification
            # Arms raised
            if (left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y):
                return "arms_raised"
            # Arms spread
            elif (abs(left_wrist.x - left_shoulder.x) > 0.3 and 
                  abs(right_wrist.x - right_shoulder.x) > 0.3):
                return "arms_spread"
            else:
                return "neutral"
                
        except:
            return "unknown"
    
    async def _analyze_temporal(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze temporal patterns in video\"\"\"
        try:
            logger.info("Starting temporal analysis...")
            
            # Analyze frame-to-frame changes
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            prev_frame = None
            motion_timeline = []
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_frame is not None:
                    # Calculate motion/change
                    diff = cv2.absdiff(prev_frame, gray)
                    motion_score = np.mean(diff) / 255.0
                    
                    timestamp = frame_number / fps
                    motion_timeline.append({
                        "timestamp": timestamp,
                        "motion_score": motion_score
                    })
                
                prev_frame = gray
                frame_number += 1
            
            cap.release()
            
            # Analyze motion patterns
            motion_scores = [item['motion_score'] for item in motion_timeline]
            
            # Find motion peaks (high activity periods)
            motion_threshold = np.mean(motion_scores) + np.std(motion_scores)
            high_motion_periods = []
            
            for item in motion_timeline:
                if item['motion_score'] > motion_threshold:
                    high_motion_periods.append(item)
            
            return {
                "motion_timeline": motion_timeline,
                "statistics": {
                    "avg_motion": np.mean(motion_scores) if motion_scores else 0,
                    "max_motion": np.max(motion_scores) if motion_scores else 0,
                    "motion_variance": np.var(motion_scores) if motion_scores else 0
                },
                "high_motion_periods": high_motion_periods,
                "motion_threshold": motion_threshold
            }
            
        except Exception as e:
            logger.error(f"Temporal analysis error: {str(e)}")
            raise
    
    async def _analyze_audio(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze audio characteristics\"\"\"
        try:
            logger.info("Starting audio analysis...")
            
            # Extract audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_path = temp_audio.name
            
            try:
                video = VideoFileClip(video_path)
                audio = video.audio
                audio.write_audiofile(audio_path, verbose=False, logger=None)
                video.close()
                
                # Load audio with librosa
                y, sr = librosa.load(audio_path)
                
                # Audio features
                duration = librosa.get_duration(y=y, sr=sr)
                
                # Spectral features
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                
                # Tempo
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                
                # Zero crossing rate
                zcr = librosa.feature.zero_crossing_rate(y)[0]
                
                # RMS energy
                rms = librosa.feature.rms(y=y)[0]
                
                # Silence detection
                silence_threshold = np.max(rms) * 0.1
                silence_ratio = np.sum(rms < silence_threshold) / len(rms)
                
                return {
                    "duration": duration,
                    "sample_rate": sr,
                    "tempo": float(tempo),
                    "spectral_centroid": {
                        "mean": float(np.mean(spectral_centroids)),
                        "std": float(np.std(spectral_centroids))
                    },
                    "spectral_rolloff": {
                        "mean": float(np.mean(spectral_rolloff)),
                        "std": float(np.std(spectral_rolloff))
                    },
                    "zero_crossing_rate": {
                        "mean": float(np.mean(zcr)),
                        "std": float(np.std(zcr))
                    },
                    "energy": {
                        "mean": float(np.mean(rms)),
                        "max": float(np.max(rms))
                    },
                    "silence_ratio": float(silence_ratio),
                    "beats_count": len(beats)
                }
                
            finally:
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
        except Exception as e:
            logger.error(f"Audio analysis error: {str(e)}")
            raise
    
    async def _generate_thumbnails(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Generate thumbnails at different timestamps\"\"\"
        try:
            logger.info("Generating thumbnails...")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            num_thumbnails = config.get('num_thumbnails', 5)
            width = config.get('width', 320)
            height = config.get('height', 240)
            
            thumbnails = []
            
            # Generate thumbnails at regular intervals
            for i in range(num_thumbnails):
                timestamp = (i + 1) * duration / (num_thumbnails + 1)
                frame_number = int(timestamp * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                
                if ret:
                    # Resize frame
                    thumbnail = cv2.resize(frame, (width, height))
                    
                    # Convert to RGB and then to base64
                    thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(thumbnail_rgb)
                    
                    # Convert to base64
                    import io
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    thumbnails.append({
                        "timestamp": timestamp,
                        "frame_number": frame_number,
                        "image": img_str,
                        "width": width,
                        "height": height
                    })
            
            cap.release()
            
            return {
                "thumbnails": thumbnails,
                "total_thumbnails": len(thumbnails),
                "dimensions": {"width": width, "height": height}
            }
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}")
            raise
    
    async def _detect_highlights(self, video_path: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Detect highlight moments in video\"\"\"
        try:
            logger.info("Starting highlight detection...")
            
            # Combine multiple analysis methods for highlight detection
            
            # 1. Motion analysis for high activity moments
            temporal_result = await self._analyze_temporal(video_path, {})
            motion_timeline = temporal_result['motion_timeline']
            
            # 2. Audio analysis for loud/interesting moments
            audio_result = await self._analyze_audio(video_path, {})
            
            # 3. Scene changes
            scene_result = await self._detect_scenes(video_path, {})
            
            # Find highlights based on multiple criteria
            highlights = []
            
            # High motion moments
            motion_scores = [item['motion_score'] for item in motion_timeline]
            motion_threshold = np.mean(motion_scores) + 1.5 * np.std(motion_scores)
            
            for item in motion_timeline:
                if item['motion_score'] > motion_threshold:
                    highlights.append({
                        "timestamp": item['timestamp'],
                        "type": "high_motion",
                        "score": item['motion_score'],
                        "reason": "High activity detected"
                    })
            
            # Scene transitions as potential highlights
            for scene in scene_result['results']['scenes'][:5]:  # Top 5 scenes
                highlights.append({
                    "timestamp": scene['start_time'],
                    "type": "scene_change",
                    "score": 0.8,
                    "reason": "Scene transition"
                })
            
            # Sort by score and remove duplicates
            highlights.sort(key=lambda x: x['score'], reverse=True)
            
            # Remove highlights too close to each other
            filtered_highlights = []
            min_interval = config.get('min_highlight_interval', 10)  # seconds
            
            for highlight in highlights:
                is_too_close = False
                for existing in filtered_highlights:
                    if abs(highlight['timestamp'] - existing['timestamp']) < min_interval:
                        is_too_close = True
                        break
                
                if not is_too_close:
                    filtered_highlights.append(highlight)
            
            return {
                "highlights": filtered_highlights[:config.get('max_highlights', 10)],
                "total_highlights": len(filtered_highlights),
                "detection_criteria": ["high_motion", "scene_changes"],
                "min_interval": min_interval
            }
            
        except Exception as e:
            logger.error(f"Highlight detection error: {str(e)}")
            raise
    
    async def extract_frames(self, video_path: str, frame_interval: int = 1, max_frames: int = 100) -> List[str]:
        \"\"\"Extract frames from video as base64 encoded strings\"\"\"
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            frames = []
            frame_number = 0
            extracted_count = 0
            
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_interval == 0:
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    
                    # Convert to base64
                    import io
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='JPEG')
                    img_str = base64.b64encode(buffer.getvalue()).decode()
                    
                    timestamp = frame_number / fps
                    frames.append({
                        "frame_number": frame_number,
                        "timestamp": timestamp,
                        "image": img_str
                    })
                    extracted_count += 1
                
                frame_number += 1
            
            cap.release()
            return frames
            
        except Exception as e:
            logger.error(f"Frame extraction error: {str(e)}")
            raise
    
    async def generate_thumbnail(self, video_path: str, timestamp: float = 0.0, width: int = 320, height: int = 240) -> str:
        \"\"\"Generate a single thumbnail at specified timestamp\"\"\"
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Seek to timestamp
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                raise ValueError("Could not extract frame at specified timestamp")
            
            # Resize and convert
            thumbnail = cv2.resize(frame, (width, height))
            thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(thumbnail_rgb)
            
            # Convert to base64
            import io
            buffer = io.BytesIO()
            pil_image.save(buffer, format='JPEG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Thumbnail generation error: {str(e)}")
            raise
    
    def _get_model_version(self, analysis_type: str) -> str:
        \"\"\"Get model version for specific analysis type\"\"\"
        model_versions = {
            "transcription": "Whisper-base",
            "objects_tracking": "YOLOv8n",
            "face_tracking": "MediaPipe-Face",
            "action_recognition": "MediaPipe-Pose",
            "scene_detection": "PySceneDetect"
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
            "ffmpeg_available": self.ffmpeg_available,
            "available_analyses": [
                "transcription", "objects_tracking", "scene_detection",
                "face_tracking", "action_recognition", "temporal_analysis",
                "audio_analysis", "thumbnail_generation", "highlight_detection"
            ],
            "loaded_models": list(self.models.keys()),
            "memory_usage": self.get_memory_usage()
        }
