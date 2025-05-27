import asyncio
import time
import psutil
from typing import Dict, List, Any, Optional
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    pipeline, AutoModel
)
import torch
from sentence_transformers import SentenceTransformer
import spacy
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from loguru import logger

class TextAnalyzer:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        asyncio.create_task(self._load_models())
    
    async def _load_models(self):
        \"\"\"Load all required models\"\"\"
        try:
            logger.info("Loading NLP models...")
            
            # Sentiment Analysis
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            
            # Emotion Detection
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if self.device == "cuda" else -1
            )
            
            # Named Entity Recognition
            self.models['ner'] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1
            )
            
            # Text Summarization
            self.models['summarization'] = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if self.device == "cuda" else -1
            )
            
            # Sentence Embeddings for similarity
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Text Classification
            self.models['classification'] = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if self.device == "cuda" else -1
            )
            
            # Load spaCy model for advanced NLP
            try:
                self.models['spacy'] = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Some features will be limited.")
                self.models['spacy'] = None
            
            # Topic modeling (will be initialized per request)
            self.models['topic_vectorizer'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english'
            )
            
            self.models_loaded = True
            logger.info("All NLP models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    async def analyze(self, text: str, analysis_type: str, config: Dict = None, language: str = "auto") -> Dict[str, Any]:
        \"\"\"Main analysis method\"\"\"
        if not self.models_loaded:
            raise ValueError("Models not loaded yet. Please wait.")
        
        config = config or {}
        start_time = time.time()
        
        try:
            if analysis_type == "sentiment":
                result = await self._analyze_sentiment(text, config)
            elif analysis_type == "entities":
                result = await self._extract_entities(text, config)
            elif analysis_type == "topics":
                result = await self._analyze_topics([text], config)
            elif analysis_type == "summary":
                result = await self._summarize_text(text, config)
            elif analysis_type == "emotions":
                result = await self._analyze_emotions(text, config)
            elif analysis_type == "classification":
                result = await self._classify_text(text, config)
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
    
    async def _analyze_sentiment(self, text: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze sentiment of text\"\"\"
        try:
            # Use transformer model
            result = self.models['sentiment'](text)
            primary_result = result[0]
            
            # Also use TextBlob for comparison
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            return {
                "label": primary_result['label'],
                "confidence": primary_result['score'],
                "polarity": textblob_polarity,  # -1 (negative) to 1 (positive)
                "subjectivity": textblob_subjectivity,  # 0 (objective) to 1 (subjective)
                "interpretation": self._interpret_sentiment(primary_result['label'], primary_result['score'])
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            raise
    
    async def _extract_entities(self, text: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Extract named entities from text\"\"\"
        try:
            # Use transformer-based NER
            ner_results = self.models['ner'](text)
            
            entities = []
            for entity in ner_results:
                entities.append({
                    "text": entity['word'],
                    "label": entity['entity_group'],
                    "confidence": entity['score'],
                    "start": entity['start'],
                    "end": entity['end']
                })
            
            # Use spaCy if available for additional processing
            spacy_entities = []
            if self.models['spacy']:
                doc = self.models['spacy'](text)
                for ent in doc.ents:
                    spacy_entities.append({
                        "text": ent.text,
                        "label": ent.label_,
                        "description": spacy.explain(ent.label_),
                        "start": ent.start_char,
                        "end": ent.end_char
                    })
            
            return {
                "entities": entities,
                "spacy_entities": spacy_entities,
                "total_entities": len(entities),
                "entity_types": list(set([e['label'] for e in entities]))
            }
        except Exception as e:
            logger.error(f"Entity extraction error: {str(e)}")
            raise
    
    async def _analyze_topics(self, texts: List[str], config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze topics in text(s)\"\"\"
        try:
            num_topics = config.get('num_topics', 5)
            
            # Vectorize texts
            tfidf_matrix = self.models['topic_vectorizer'].fit_transform(texts)
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                max_iter=10
            )
            lda.fit(tfidf_matrix)
            
            # Get feature names
            feature_names = self.models['topic_vectorizer'].get_feature_names_out()
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "weights": top_weights.tolist()
                })
            
            return {
                "topics": topics,
                "num_topics": num_topics,
                "coherence_score": 0.5  # Placeholder
            }
        except Exception as e:
            logger.error(f"Topic analysis error: {str(e)}")
            raise
    
    async def _summarize_text(self, text: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Summarize text\"\"\"
        try:
            max_length = config.get('max_length', 150)
            min_length = config.get('min_length', 50)
            
            # Check if text is long enough to summarize
            if len(text.split()) < min_length:
                return {
                    "summary": text,
                    "original_length": len(text.split()),
                    "summary_length": len(text.split()),
                    "compression_ratio": 1.0,
                    "note": "Text too short to summarize"
                }
            
            summary_result = self.models['summarization'](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            summary = summary_result[0]['summary_text']
            
            return {
                "summary": summary,
                "original_length": len(text.split()),
                "summary_length": len(summary.split()),
                "compression_ratio": len(summary.split()) / len(text.split())
            }
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            raise
    
    async def _analyze_emotions(self, text: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Analyze emotions in text\"\"\"
        try:
            emotion_results = self.models['emotion'](text)
            
            emotions = []
            for emotion in emotion_results:
                emotions.append({
                    "emotion": emotion['label'],
                    "confidence": emotion['score']
                })
            
            # Sort by confidence
            emotions.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "emotions": emotions,
                "dominant_emotion": emotions[0]['emotion'] if emotions else None,
                "emotional_intensity": emotions[0]['confidence'] if emotions else 0
            }
        except Exception as e:
            logger.error(f"Emotion analysis error: {str(e)}")
            raise
    
    async def _classify_text(self, text: str, config: Dict) -> Dict[str, Any]:
        \"\"\"Classify text into categories\"\"\"
        try:
            categories = config.get('categories', [
                'technology', 'business', 'sports', 'entertainment', 
                'politics', 'health', 'science', 'other'
            ])
            
            classification_result = self.models['classification'](text, categories)
            
            return {
                "predictions": [
                    {
                        "label": label,
                        "confidence": score
                    }
                    for label, score in zip(
                        classification_result['labels'],
                        classification_result['scores']
                    )
                ],
                "top_category": classification_result['labels'][0],
                "confidence": classification_result['scores'][0]
            }
        except Exception as e:
            logger.error(f"Classification error: {str(e)}")
            raise
    
    async def compute_similarity(self, text1: str, text2: str) -> float:
        \"\"\"Compute semantic similarity between two texts\"\"\"
        try:
            embeddings = self.models['sentence_transformer'].encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception as e:
            logger.error(f"Similarity computation error: {str(e)}")
            raise
    
    def interpret_similarity(self, score: float) -> str:
        \"\"\"Interpret similarity score\"\"\"
        if score > 0.8:
            return "Very similar"
        elif score > 0.6:
            return "Similar"
        elif score > 0.4:
            return "Somewhat similar"
        elif score > 0.2:
            return "Slightly similar"
        else:
            return "Not similar"
    
    def _interpret_sentiment(self, label: str, confidence: float) -> str:
        \"\"\"Interpret sentiment results\"\"\"
        intensity = "strong" if confidence > 0.8 else "moderate" if confidence > 0.6 else "weak"
        return f"{intensity} {label.lower()}"
    
    def _get_model_version(self, analysis_type: str) -> str:
        \"\"\"Get model version for specific analysis type\"\"\"
        model_versions = {
            "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "emotions": "j-hartmann/emotion-english-distilroberta-base",
            "entities": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "summary": "facebook/bart-large-cnn",
            "classification": "facebook/bart-large-mnli"
        }
        return model_versions.get(analysis_type, "unknown")
    
    def get_memory_usage(self) -> Dict[str, float]:
        \"\"\"Get current memory usage\"\"\"
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "cpu_percent": process.cpu_percent()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        \"\"\"Get information about loaded models\"\"\"
        return {
            "models_loaded": self.models_loaded,
            "device": self.device,
            "available_analyses": [
                "sentiment", "entities", "topics", 
                "summary", "emotions", "classification"
            ],
            "loaded_models": list(self.models.keys()),
            "memory_usage": self.get_memory_usage()
        }
