# Create multimodal fusion engine.py
import asyncio
import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
import json
import re
from collections import defaultdict, Counter

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Some dimensionality reduction features will be limited.")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available. Some clustering features will be limited.")

class MultimodalFusionEngine:
    def __init__(self):
        self.models = {}
        self.models_loaded = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Multimodal Fusion Engine on {self.device}")
        
        # Initialize fusion algorithms
        self.fusion_algorithms = {
            "semantic": self._semantic_fusion,
            "temporal": self._temporal_fusion, 
            "cross_modal": self._cross_modal_fusion,
            "statistical": self._statistical_fusion,
            "graph_based": self._graph_based_fusion
        }
        
        # Initialize models
        asyncio.create_task(self._load_models())
    
    async def _load_models(self):
        # \"\"\"Load models required for multimodal fusion\"\"\"
        try:
            logger.info("Loading multimodal fusion models...")
            
            # Sentence transformer for semantic similarity
            logger.info("Loading sentence transformer...")
            self.models['sentence_transformer'] = SentenceTransformer(
                'all-MiniLM-L6-v2',
                device=self.device
            )
            
            # Load universal sentence encoder alternative if needed
            self.models['text_encoder'] = self.models['sentence_transformer']
            
            self.models_loaded = True
            logger.info("All multimodal fusion models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            self.models_loaded = False
    
    async def fuse_modalities(
        self, 
        analysis_results: Dict[str, Any], 
        fusion_types: List[str], 
        config: Dict = None
    ) -> Dict[str, Any]:
        # \"\"\"Main fusion method that combines analysis from multiple modalities\"\"\"
        if not self.models_loaded:
            raise ValueError("Models not loaded yet. Please wait.")
        
        config = config or {}
        start_time = time.time()
        
        try:
            logger.info(f"Starting multimodal fusion with types: {fusion_types}")
            
            # Prepare data for fusion
            prepared_data = self._prepare_fusion_data(analysis_results)
            
            # Apply each fusion type
            fusion_results = {}
            
            for fusion_type in fusion_types:
                if fusion_type in self.fusion_algorithms:
                    logger.info(f"Applying {fusion_type} fusion...")
                    result = await self.fusion_algorithms[fusion_type](
                        prepared_data, config
                    )
                    fusion_results[fusion_type] = result
                else:
                    logger.warning(f"Unknown fusion type: {fusion_type}")
            
            # Combine all fusion results
            combined_results = self._combine_fusion_results(fusion_results)
            
            processing_time = time.time() - start_time
            
            return {
                "fusion_type": "multimodal",
                "results": combined_results,
                "confidence": combined_results.get("overall_confidence", 0.7),
                "processing_time": processing_time,
                "fusion_types_applied": fusion_types
            }
            
        except Exception as e:
            logger.error(f"Multimodal fusion error: {str(e)}")
            raise
    
    def _prepare_fusion_data(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Prepare and normalize data from different modalities for fusion\"\"\"
        prepared = {
            "text_data": [],
            "image_data": [],
            "video_data": [],
            "temporal_data": [],
            "entities": [],
            "sentiments": [],
            "embeddings": {}
        }
        
        # Process text analysis results
        if "text" in analysis_results:
            text_results = analysis_results["text"]
            
            # Extract text content and embeddings
            if "results" in text_results:
                text_res = text_results["results"]
                
                # Sentiment data
                if "sentiment" in text_res:
                    prepared["sentiments"].append({
                        "modality": "text",
                        "sentiment": text_res["sentiment"],
                        "confidence": text_res["sentiment"].get("confidence", 0.5)
                    })
                
                # Entity data
                if "entities" in text_res:
                    for entity in text_res["entities"].get("entities", []):
                        prepared["entities"].append({
                            "modality": "text",
                            "text": entity["text"],
                            "label": entity["label"],
                            "confidence": entity["confidence"]
                        })
                
                # Text content for embedding
                if "extracted_text" in text_res:
                    prepared["text_data"].append(text_res["extracted_text"])
        
        # Process image analysis results
        if "image" in analysis_results:
            image_results = analysis_results["image"]
            
            if "results" in image_results:
                img_res = image_results["results"]
                
                # Object detection data
                if "objects" in img_res:
                    for obj in img_res["objects"].get("objects", []):
                        prepared["entities"].append({
                            "modality": "image",
                            "text": obj["class"],
                            "label": "OBJECT",
                            "confidence": obj["confidence"]
                        })
                
                # OCR text data
                if "ocr" in img_res:
                    ocr_text = img_res["ocr"].get("extracted_text", "")
                    if ocr_text:
                        prepared["text_data"].append(ocr_text)
                
                # Scene classification
                if "scenes" in img_res:
                    scene_info = img_res["scenes"]
                    if "top_scene" in scene_info:
                        prepared["entities"].append({
                            "modality": "image",
                            "text": scene_info["top_scene"],
                            "label": "SCENE",
                            "confidence": scene_info.get("confidence", 0.5)
                        })
        
        # Process video analysis results
        if "video" in analysis_results:
            video_results = analysis_results["video"]
            
            if "results" in video_results:
                vid_res = video_results["results"]
                
                # Transcription data
                if "transcription" in vid_res:
                    transcript = vid_res["transcription"]
                    if "transcript" in transcript:
                        prepared["text_data"].append(transcript["transcript"])
                    
                    # Temporal segments
                    if "segments" in transcript:
                        for segment in transcript["segments"]:
                            prepared["temporal_data"].append({
                                "modality": "video",
                                "start_time": segment["start"],
                                "end_time": segment["end"],
                                "content": segment["text"],
                                "confidence": segment.get("confidence", 0.5)
                            })
                
                # Object tracking data
                if "objects_tracking" in vid_res:
                    tracking_data = vid_res["objects_tracking"]
                    if "object_statistics" in tracking_data:
                        for obj_class, stats in tracking_data["object_statistics"].items():
                            prepared["entities"].append({
                                "modality": "video",
                                "text": obj_class,
                                "label": "TRACKED_OBJECT",
                                "confidence": stats.get("avg_confidence", 0.5),
                                "temporal_info": {
                                    "first_seen": stats.get("first_seen", 0),
                                    "last_seen": stats.get("last_seen", 0),
                                    "frequency": stats.get("count", 0)
                                }
                            })
        
        return prepared
    
    async def _semantic_fusion(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        # \"\"\"Perform semantic fusion across modalities\"\"\"
        try:
            logger.info("Performing semantic fusion...")
            
            # Collect all text content
            all_texts = data["text_data"]
            
            if not all_texts:
                return {"semantic_similarity": {}, "clusters": [], "topics": []}
            
            # Generate embeddings
            embeddings = self.models['text_encoder'].encode(all_texts)
            
            # Calculate semantic similarity matrix
            similarity_matrix = cosine_similarity(embeddings)
            
            # Perform clustering if we have enough texts
            clusters = []
            if len(all_texts) > 2:
                # Use DBSCAN for clustering
                clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine')
                cluster_labels = clustering.fit_predict(embeddings)
                
                # Organize clusters
                cluster_dict = defaultdict(list)
                for i, label in enumerate(cluster_labels):
                    if label != -1:  # -1 is noise in DBSCAN
                        cluster_dict[label].append({
                            "text": all_texts[i],
                            "index": i
                        })
                
                clusters = [{"cluster_id": k, "texts": v} for k, v in cluster_dict.items()]
            
            # Extract semantic topics (simplified)
            topics = self._extract_semantic_topics(all_texts, embeddings)
            
            # Calculate cross-modal semantic alignment
            entity_alignment = self._align_entities_semantically(data["entities"])
            
            return {
                "semantic_similarity_matrix": similarity_matrix.tolist(),
                "text_clusters": clusters,
                "semantic_topics": topics,
                "entity_alignment": entity_alignment,
                "overall_coherence": float(np.mean(similarity_matrix))
            }
            
        except Exception as e:
            logger.error(f"Semantic fusion error: {str(e)}")
            raise
    
    async def _temporal_fusion(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        # \"\"\"Perform temporal fusion to align time-based events\"\"\"
        try:
            logger.info("Performing temporal fusion...")
            
            temporal_events = data["temporal_data"]
            
            if not temporal_events:
                return {"timeline": [], "synchronization_score": 0.0}
            
            # Sort events by start time
            sorted_events = sorted(temporal_events, key=lambda x: x["start_time"])
            
            # Create unified timeline
            timeline = []
            for event in sorted_events:
                timeline.append({
                    "start_time": event["start_time"],
                    "end_time": event["end_time"],
                    "duration": event["end_time"] - event["start_time"],
                    "modality": event["modality"],
                    "content": event["content"],
                    "confidence": event["confidence"]
                })
            
            # Detect temporal overlaps and synchronization points
            overlaps = self._detect_temporal_overlaps(timeline)
            
            # Calculate synchronization quality
            sync_score = self._calculate_synchronization_score(timeline)
            
            # Identify key temporal moments
            key_moments = self._identify_key_moments(timeline)
            
            return {
                "unified_timeline": timeline,
                "temporal_overlaps": overlaps,
                "synchronization_score": sync_score,
                "key_moments": key_moments,
                "total_duration": max([e["end_time"] for e in timeline]) if timeline else 0
            }
            
        except Exception as e:
            logger.error(f"Temporal fusion error: {str(e)}")
            raise
    
    async def _cross_modal_fusion(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        # \"\"\"Perform cross-modal fusion to find relationships between modalities\"\"\"
        try:
            logger.info("Performing cross-modal fusion...")
            
            # Analyze entity correspondences across modalities
            entity_correspondences = self._find_entity_correspondences(data["entities"])
            
            # Analyze sentiment consistency
            sentiment_consistency = self._analyze_sentiment_consistency(data["sentiments"])
            
            # Create cross-modal relationship graph
            relationship_graph = self._build_relationship_graph(data)
            
            # Calculate cross-modal confidence
            cross_modal_confidence = self._calculate_cross_modal_confidence(
                entity_correspondences, sentiment_consistency
            )
            
            return {
                "entity_correspondences": entity_correspondences,
                "sentiment_consistency": sentiment_consistency,
                "relationship_graph": relationship_graph,
                "cross_modal_confidence": cross_modal_confidence,
                "modality_agreement_score": self._calculate_modality_agreement(data)
            }
            
        except Exception as e:
            logger.error(f"Cross-modal fusion error: {str(e)}")
            raise
    
    async def _statistical_fusion(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        # \"\"\"Perform statistical fusion to identify patterns and anomalies\"\"\"
        try:
            logger.info("Performing statistical fusion...")
            
            # Collect numerical features from all modalities
            features = self._extract_numerical_features(data)
            
            if not features:
                return {"statistics": {}, "patterns": [], "anomalies": []}
            
            # Perform statistical analysis
            stats_summary = {}
            for feature_name, values in features.items():
                if values:
                    stats_summary[feature_name] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values))
                    }
            
            # Detect statistical patterns
            patterns = self._detect_statistical_patterns(features)
            
            # Detect anomalies
            anomalies = self._detect_statistical_anomalies(features)
            
            # Correlation analysis between features
            correlations = self._calculate_feature_correlations(features)
            
            return {
                "feature_statistics": stats_summary,
                "detected_patterns": patterns,
                "detected_anomalies": anomalies,
                "feature_correlations": correlations
            }
            
        except Exception as e:
            logger.error(f"Statistical fusion error: {str(e)}")
            raise
    
    async def _graph_based_fusion(self, data: Dict[str, Any], config: Dict) -> Dict[str, Any]:
        # \"\"\"Perform graph-based fusion using network analysis\"\"\"
        try:
            logger.info("Performing graph-based fusion...")
            
            # Build multimodal knowledge graph
            G = nx.Graph()
            
            # Add entity nodes
            for entity in data["entities"]:
                node_id = f"{entity['modality']}_{entity['text']}"
                G.add_node(node_id, 
                          text=entity["text"],
                          label=entity["label"],
                          modality=entity["modality"],
                          confidence=entity["confidence"])
            
            # Add edges based on semantic similarity and co-occurrence
            self._add_semantic_edges(G, data["entities"])
            
            # Calculate graph metrics
            centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            
            # Detect communities
            communities = list(nx.community.greedy_modularity_communities(G))
            
            # Find key nodes (high centrality)
            key_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "graph_nodes": len(G.nodes()),
                "graph_edges": len(G.edges()),
                "centrality_scores": centrality,
                "betweenness_scores": betweenness,
                "communities": [list(community) for community in communities],
                "key_entities": key_nodes,
                "graph_density": nx.density(G)
            }
            
        except Exception as e:
            logger.error(f"Graph-based fusion error: {str(e)}")
            raise
    
    def _combine_fusion_results(self, fusion_results: Dict[str, Dict]) -> Dict[str, Any]:
        # \"\"\"Combine results from different fusion algorithms\"\"\"
        combined = {
            "fusion_summary": {},
            "cross_modal_insights": {},
            "unified_entities": [],
            "temporal_alignment": {},
            "statistical_insights": {},
            "graph_insights": {}
        }
        
        # Combine semantic fusion results
        if "semantic" in fusion_results:
            semantic = fusion_results["semantic"]
            combined["fusion_summary"]["semantic_coherence"] = semantic.get("overall_coherence", 0.0)
            combined["cross_modal_insights"]["topics"] = semantic.get("semantic_topics", [])
            combined["cross_modal_insights"]["entity_alignment"] = semantic.get("entity_alignment", {})
        
        # Combine temporal fusion results
        if "temporal" in fusion_results:
            temporal = fusion_results["temporal"]
            combined["temporal_alignment"] = temporal
            combined["fusion_summary"]["temporal_sync_score"] = temporal.get("synchronization_score", 0.0)
        
        # Combine cross-modal fusion results
        if "cross_modal" in fusion_results:
            cross_modal = fusion_results["cross_modal"]
            combined["cross_modal_insights"].update(cross_modal)
            combined["fusion_summary"]["cross_modal_confidence"] = cross_modal.get("cross_modal_confidence", 0.0)
        
        # Combine statistical fusion results
        if "statistical" in fusion_results:
            statistical = fusion_results["statistical"]
            combined["statistical_insights"] = statistical
        
        # Combine graph-based fusion results
        if "graph_based" in fusion_results:
            graph = fusion_results["graph_based"]
            combined["graph_insights"] = graph
        
        # Calculate overall confidence
        confidences = []
        if "semantic_coherence" in combined["fusion_summary"]:
            confidences.append(combined["fusion_summary"]["semantic_coherence"])
        if "cross_modal_confidence" in combined["fusion_summary"]:
            confidences.append(combined["fusion_summary"]["cross_modal_confidence"])
        if "temporal_sync_score" in combined["fusion_summary"]:
            confidences.append(combined["fusion_summary"]["temporal_sync_score"])
        
        combined["overall_confidence"] = float(np.mean(confidences)) if confidences else 0.5
        
        return combined
    
    def _extract_semantic_topics(self, texts: List[str], embeddings: np.ndarray) -> List[Dict[str, Any]]:
        # \"\"\"Extract semantic topics from text embeddings\"\"\"
        try:
            if len(texts) < 3:
                return []
            
            # Use K-means for topic clustering
            n_topics = min(3, len(texts) // 2)
            kmeans = KMeans(n_clusters=n_topics, random_state=42, n_init=10)
            topic_labels = kmeans.fit_predict(embeddings)
            
            # Organize topics
            topics = []
            for topic_id in range(n_topics):
                topic_texts = [texts[i] for i, label in enumerate(topic_labels) if label == topic_id]
                if topic_texts:
                    # Extract key terms (simplified)
                    word_freq = Counter()
                    for text in topic_texts:
                        words = re.findall(r'\b\w+\b', text.lower())
                        word_freq.update(words)
                    
                    top_words = [word for word, freq in word_freq.most_common(5)]
                    
                    topics.append({
                        "topic_id": topic_id,
                        "texts": topic_texts,
                        "key_terms": top_words,
                        "size": len(topic_texts)
                    })
            
            return topics
            
        except Exception as e:
            logger.error(f"Topic extraction error: {str(e)}")
            return []
    
    def _align_entities_semantically(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        # \"\"\"Align entities across modalities based on semantic similarity\"\"\"
        try:
            if not entities:
                return {}
            
            # Group entities by modality
            modality_entities = defaultdict(list)
            for entity in entities:
                modality_entities[entity["modality"]].append(entity)
            
            # Find potential alignments
            alignments = []
            modalities = list(modality_entities.keys())
            
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    mod1, mod2 = modalities[i], modalities[j]
                    
                    for ent1 in modality_entities[mod1]:
                        for ent2 in modality_entities[mod2]:
                            # Simple string similarity
                            similarity = self._calculate_string_similarity(
                                ent1["text"], ent2["text"]
                            )
                            
                            if similarity > 0.7:  # Threshold for alignment
                                alignments.append({
                                    "entity1": ent1,
                                    "entity2": ent2,
                                    "similarity": similarity,
                                    "modalities": [mod1, mod2]
                                })
            
            return {
                "alignments": alignments,
                "alignment_count": len(alignments),
                "modalities_involved": modalities
            }
            
        except Exception as e:
            logger.error(f"Entity alignment error: {str(e)}")
            return {}
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        # \"\"\"Calculate similarity between two strings\"\"\"
        from difflib import SequenceMatcher
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
    
    def _detect_temporal_overlaps(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # \"\"\"Detect overlapping events in timeline\"\"\"
        overlaps = []
        
        for i in range(len(timeline)):
            for j in range(i + 1, len(timeline)):
                event1, event2 = timeline[i], timeline[j]
                
                # Check for temporal overlap
                if (event1["start_time"] <= event2["end_time"] and 
                    event2["start_time"] <= event1["end_time"]):
                    
                    overlap_start = max(event1["start_time"], event2["start_time"])
                    overlap_end = min(event1["end_time"], event2["end_time"])
                    
                    overlaps.append({
                        "event1_index": i,
                        "event2_index": j,
                        "overlap_start": overlap_start,
                        "overlap_end": overlap_end,
                        "overlap_duration": overlap_end - overlap_start,
                        "modalities": [event1["modality"], event2["modality"]]
                    })
        
        return overlaps
    
    def _calculate_synchronization_score(self, timeline: List[Dict[str, Any]]) -> float:
        # \"\"\"Calculate how well events are synchronized across modalities\"\"\"
        if len(timeline) < 2:
            return 1.0
        
        # Calculate time gaps between consecutive events
        gaps = []
        for i in range(1, len(timeline)):
            gap = timeline[i]["start_time"] - timeline[i-1]["end_time"]
            gaps.append(abs(gap))
        
        # Lower variance in gaps indicates better synchronization
        if gaps:
            gap_variance = np.var(gaps)
            # Normalize to 0-1 score (lower variance = higher score)
            sync_score = 1.0 / (1.0 + gap_variance)
            return float(sync_score)
        
        return 1.0
    
    def _identify_key_moments(self, timeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # \"\"\"Identify key moments in the timeline\"\"\"
        key_moments = []
        
        # Find moments with high confidence
        high_confidence_threshold = 0.8
        for event in timeline:
            if event["confidence"] >= high_confidence_threshold:
                key_moments.append({
                    "timestamp": event["start_time"],
                    "reason": "high_confidence",
                    "confidence": event["confidence"],
                    "content": event["content"][:100] + "..." if len(event["content"]) > 100 else event["content"]
                })
        
        # Find moments with multiple modalities
        time_bins = defaultdict(list)
        for event in timeline:
            time_bin = int(event["start_time"] // 5) * 5  # 5-second bins
            time_bins[time_bin].append(event)
        
        for time_bin, events in time_bins.items():
            if len(events) > 1:
                modalities = set(event["modality"] for event in events)
                if len(modalities) > 1:
                    key_moments.append({
                        "timestamp": time_bin,
                        "reason": "multi_modal_convergence",
                        "modalities": list(modalities),
                        "event_count": len(events)
                    })
        
        return key_moments
    
    def _find_entity_correspondences(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        # \"\"\"Find corresponding entities across modalities\"\"\"
        correspondences = []
        
        # Group by entity text (case-insensitive)
        entity_groups = defaultdict(list)
        for entity in entities:
            key = entity["text"].lower().strip()
            entity_groups[key].append(entity)
        
        # Find entities that appear in multiple modalities
        for entity_text, entity_list in entity_groups.items():
            modalities = set(entity["modality"] for entity in entity_list)
            if len(modalities) > 1:
                correspondences.append({
                    "entity_text": entity_text,
                    "modalities": list(modalities),
                    "occurrences": len(entity_list),
                    "avg_confidence": np.mean([entity["confidence"] for entity in entity_list]),
                    "entities": entity_list
                })
        
        return {
            "correspondences": correspondences,
            "total_correspondences": len(correspondences),
            "cross_modal_entities": [c["entity_text"] for c in correspondences]
        }
    
    def _analyze_sentiment_consistency(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        # \"\"\"Analyze sentiment consistency across modalities\"\"\"
        if not sentiments:
            return {"consistency_score": 1.0, "sentiments": []}
        
        # Extract sentiment labels and scores
        sentiment_data = []
        for sent in sentiments:
            if "sentiment" in sent and "label" in sent["sentiment"]:
                sentiment_data.append({
                    "modality": sent["modality"],
                    "label": sent["sentiment"]["label"],
                    "confidence": sent["confidence"]
                })
        
        if not sentiment_data:
            return {"consistency_score": 1.0, "sentiments": []}
        
        # Calculate consistency
        labels = [s["label"] for s in sentiment_data]
        unique_labels = set(labels)
        
        if len(unique_labels) == 1:
            consistency_score = 1.0
        else:
            # Calculate based on label distribution
            label_counts = Counter(labels)
            most_common_count = label_counts.most_common(1)[0][1]
            consistency_score = most_common_count / len(labels)
        
        return {
            "consistency_score": consistency_score,
            "sentiment_distribution": dict(Counter(labels)),
            "modalities_analyzed": list(set(s["modality"] for s in sentiment_data)),
            "dominant_sentiment": Counter(labels).most_common(1)[0][0] if labels else None
        }
    
    def _build_relationship_graph(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Build a relationship graph between multimodal elements\"\"\"
        relationships = []
        
        # Create relationships between entities from different modalities
        entities = data["entities"]
        
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                ent1, ent2 = entities[i], entities[j]
                
                if ent1["modality"] != ent2["modality"]:
                    # Calculate relationship strength
                    text_sim = self._calculate_string_similarity(ent1["text"], ent2["text"])
                    
                    if text_sim > 0.5:  # Threshold for relationship
                        relationships.append({
                            "entity1": ent1["text"],
                            "entity2": ent2["text"],
                            "modality1": ent1["modality"],
                            "modality2": ent2["modality"],
                            "strength": text_sim,
                            "type": "semantic_similarity"
                        })
        
        return {
            "relationships": relationships,
            "total_relationships": len(relationships),
            "relationship_types": ["semantic_similarity"]
        }
    
    def _calculate_cross_modal_confidence(self, correspondences: Dict, sentiment_consistency: Dict) -> float:
        # \"\"\"Calculate overall cross-modal confidence\"\"\"
        factors = []
        
        # Factor 1: Entity correspondence strength
        if correspondences["total_correspondences"] > 0:
            factors.append(min(correspondences["total_correspondences"] / 5.0, 1.0))
        else:
            factors.append(0.0)
        
        # Factor 2: Sentiment consistency
        factors.append(sentiment_consistency["consistency_score"])
        
        # Average the factors
        return float(np.mean(factors)) if factors else 0.5
    
    def _calculate_modality_agreement(self, data: Dict[str, Any]) -> float:
        # \"\"\"Calculate how much modalities agree with each other\"\"\"
        # This is a simplified version
        # In a real implementation, this would involve more sophisticated analysis
        
        agreement_factors = []
        
        # Check entity agreement
        entities_by_modality = defaultdict(set)
        for entity in data["entities"]:
            entities_by_modality[entity["modality"]].add(entity["text"].lower())
        
        if len(entities_by_modality) > 1:
            modalities = list(entities_by_modality.keys())
            intersections = []
            
            for i in range(len(modalities)):
                for j in range(i + 1, len(modalities)):
                    set1 = entities_by_modality[modalities[i]]
                    set2 = entities_by_modality[modalities[j]]
                    
                    if set1 and set2:
                        intersection = len(set1.intersection(set2))
                        union = len(set1.union(set2))
                        jaccard = intersection / union if union > 0 else 0
                        intersections.append(jaccard)
            
            if intersections:
                agreement_factors.append(np.mean(intersections))
        
        return float(np.mean(agreement_factors)) if agreement_factors else 0.5
    
    def _extract_numerical_features(self, data: Dict[str, Any]) -> Dict[str, List[float]]:
        # \"\"\"Extract numerical features from multimodal data\"\"\"
        features = defaultdict(list)
        
        # Extract confidence scores
        for entity in data["entities"]:
            features["entity_confidence"].append(entity["confidence"])
            features[f"{entity['modality']}_confidence"].append(entity["confidence"])
        
        for sentiment in data["sentiments"]:
            features["sentiment_confidence"].append(sentiment["confidence"])
        
        # Extract temporal features
        for temporal in data["temporal_data"]:
            features["event_duration"].append(temporal["end_time"] - temporal["start_time"])
            features["temporal_confidence"].append(temporal["confidence"])
        
        return dict(features)
    
    def _detect_statistical_patterns(self, features: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        # \"\"\"Detect statistical patterns in numerical features\"\"\"
        patterns = []
        
        for feature_name, values in features.items():
            if len(values) > 2:
                # Check for trends
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                if abs(r_value) > 0.7:  # Strong correlation
                    pattern_type = "increasing" if slope > 0 else "decreasing"
                    patterns.append({
                        "feature": feature_name,
                        "pattern_type": f"{pattern_type}_trend",
                        "strength": abs(r_value),
                        "p_value": p_value
                    })
                
                # Check for periodicity (simplified)
                if len(values) > 10:
                    fft = np.fft.fft(values)
                    freqs = np.fft.fftfreq(len(values))
                    dominant_freq_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                    
                    if np.abs(fft[dominant_freq_idx]) > np.mean(np.abs(fft)) * 2:
                        patterns.append({
                            "feature": feature_name,
                            "pattern_type": "periodic",
                            "dominant_frequency": freqs[dominant_freq_idx],
                            "strength": float(np.abs(fft[dominant_freq_idx]) / np.mean(np.abs(fft)))
                        })
        
        return patterns
    
    def _detect_statistical_anomalies(self, features: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        # \"\"\"Detect statistical anomalies in numerical features\"\"\"
        anomalies = []
        
        for feature_name, values in features.items():
            if len(values) > 3:
                # Use IQR method for anomaly detection
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                
                if outliers:
                    anomalies.append({
                        "feature": feature_name,
                        "anomaly_type": "statistical_outlier",
                        "outlier_count": len(outliers),
                        "outlier_values": outliers,
                        "bounds": {"lower": lower_bound, "upper": upper_bound}
                    })
                
                # Z-score based anomaly detection
                z_scores = np.abs(stats.zscore(values))
                z_anomalies = [values[i] for i, z in enumerate(z_scores) if z > 2.5]
                
                if z_anomalies:
                    anomalies.append({
                        "feature": feature_name,
                        "anomaly_type": "z_score_outlier",
                        "outlier_count": len(z_anomalies),
                        "outlier_values": z_anomalies,
                        "threshold": 2.5
                    })
        
        return anomalies
    
    def _calculate_feature_correlations(self, features: Dict[str, List[float]]) -> Dict[str, Any]:
        # \"\"\"Calculate correlations between different features\"\"\"
        feature_names = list(features.keys())
        correlations = {}
        
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                feat1, feat2 = feature_names[i], feature_names[j]
                
                # Ensure same length for correlation
                len1, len2 = len(features[feat1]), len(features[feat2])
                min_len = min(len1, len2)
                
                if min_len > 2:
                    values1 = features[feat1][:min_len]
                    values2 = features[feat2][:min_len]
                    
                    corr_coef = np.corrcoef(values1, values2)[0, 1]
                    
                    if not np.isnan(corr_coef):
                        correlations[f"{feat1}_vs_{feat2}"] = {
                            "correlation": float(corr_coef),
                            "strength": "strong" if abs(corr_coef) > 0.7 else "moderate" if abs(corr_coef) > 0.4 else "weak"
                        }
        
        return correlations
    
    def _add_semantic_edges(self, G: nx.Graph, entities: List[Dict[str, Any]]):
        # \"\"\"Add edges to graph based on semantic relationships\"\"\"
        nodes = list(G.nodes())
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1, node2 = nodes[i], nodes[j]
                
                # Get node attributes
                attr1 = G.nodes[node1]
                attr2 = G.nodes[node2]
                
                # Calculate semantic similarity
                similarity = self._calculate_string_similarity(attr1["text"], attr2["text"])
                
                # Add edge if similarity is above threshold
                if similarity > 0.6:
                    G.add_edge(node1, node2, weight=similarity, type="semantic")
                
                # Add edge if same entity type but different modality
                if (attr1["label"] == attr2["label"] and 
                    attr1["modality"] != attr2["modality"]):
                    G.add_edge(node1, node2, weight=0.8, type="cross_modal")
    
    async def correlate_modalities(
        self,
        text_results: Optional[Dict[str, Any]] = None,
        image_results: Optional[Dict[str, Any]] = None,
        video_results: Optional[Dict[str, Any]] = None,
        correlation_types: List[str] = ["semantic", "entity", "sentiment", "temporal"],
        config: Dict = None
    ) -> List[Dict[str, Any]]:
        # \"\"\"Find correlations between different modalities\"\"\"
        correlations = []
        config = config or {}
        
        # Prepare analysis results for correlation
        analysis_results = {}
        if text_results:
            analysis_results["text"] = text_results
        if image_results:
            analysis_results["image"] = image_results
        if video_results:
            analysis_results["video"] = video_results
        
        prepared_data = self._prepare_fusion_data(analysis_results)
        
        for correlation_type in correlation_types:
            if correlation_type == "semantic":
                correlation = await self._correlate_semantic(prepared_data)
            elif correlation_type == "entity":
                correlation = self._correlate_entities(prepared_data)
            elif correlation_type == "sentiment":
                correlation = self._correlate_sentiment(prepared_data)
            elif correlation_type == "temporal":
                correlation = self._correlate_temporal(prepared_data)
            else:
                logger.warning(f"Unknown correlation type: {correlation_type}")
                continue
            
            correlations.append({
                "type": correlation_type,
                "correlation": correlation,
                "strength": correlation.get("strength", 0.0)
            })
        
        return correlations
    
    async def _correlate_semantic(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Correlate semantic content across modalities\"\"\"
        all_texts = data["text_data"]
        
        if len(all_texts) < 2:
            return {"strength": 0.0, "details": "Insufficient text data"}
        
        # Generate embeddings and calculate similarity
        embeddings = self.models['text_encoder'].encode(all_texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Calculate average similarity
        avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
        
        return {
            "strength": float(avg_similarity),
            "similarity_matrix": similarity_matrix.tolist(),
            "interpretation": "High semantic correlation" if avg_similarity > 0.7 else "Moderate semantic correlation" if avg_similarity > 0.4 else "Low semantic correlation"
        }
    
    def _correlate_entities(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Correlate entities across modalities\"\"\"
        entities = data["entities"]
        
        # Group entities by modality
        modality_entities = defaultdict(set)
        for entity in entities:
            modality_entities[entity["modality"]].add(entity["text"].lower())
        
        if len(modality_entities) < 2:
            return {"strength": 0.0, "details": "Insufficient modalities"}
        
        # Calculate Jaccard similarity between modalities
        modalities = list(modality_entities.keys())
        similarities = []
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                set1 = modality_entities[modalities[i]]
                set2 = modality_entities[modalities[j]]
                
                if set1 and set2:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    jaccard = intersection / union if union > 0 else 0
                    similarities.append(jaccard)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        return {
            "strength": float(avg_similarity),
            "modalities_compared": modalities,
            "common_entities": list(set.intersection(*modality_entities.values())) if modality_entities else []
        }
    
    def _correlate_sentiment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Correlate sentiment across modalities\"\"\"
        sentiments = data["sentiments"]
        
        if len(sentiments) < 2:
            return {"strength": 0.0, "details": "Insufficient sentiment data"}
        
        # Analyze sentiment consistency
        consistency = self._analyze_sentiment_consistency(sentiments)
        
        return {
            "strength": consistency["consistency_score"],
            "sentiment_distribution": consistency["sentiment_distribution"],
            "dominant_sentiment": consistency["dominant_sentiment"]
        }
    
    def _correlate_temporal(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # \"\"\"Correlate temporal patterns across modalities\"\"\"
        temporal_data = data["temporal_data"]
        
        if len(temporal_data) < 2:
            return {"strength": 0.0, "details": "Insufficient temporal data"}
        
        # Calculate temporal synchronization
        timeline = sorted(temporal_data, key=lambda x: x["start_time"])
        sync_score = self._calculate_synchronization_score(timeline)
        
        return {
            "strength": sync_score,
            "timeline_length": len(timeline),
            "temporal_overlaps": len(self._detect_temporal_overlaps(timeline))
        }
    
    async def generate_insights(
        self,
        multimodal_results: Dict[str, Any],
        insight_types: List[str] = ["patterns", "anomalies", "trends", "relationships"],
        config: Dict = None
    ) -> Dict[str, Any]:
        # \"\"\"Generate high-level insights from multimodal analysis\"\"\"
        insights = {}
        config = config or {}
        
        for insight_type in insight_types:
            if insight_type == "patterns":
                insights["patterns"] = self._identify_patterns(multimodal_results)
            elif insight_type == "anomalies":
                insights["anomalies"] = self._identify_anomalies(multimodal_results)
            elif insight_type == "trends":
                insights["trends"] = self._identify_trends(multimodal_results)
            elif insight_type == "relationships":
                insights["relationships"] = self._identify_relationships(multimodal_results)
        
        # Generate overall confidence
        insights["overall_confidence"] = self._calculate_insight_confidence(insights)
        
        return insights
    
    def _identify_patterns(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        # \"\"\"Identify patterns in multimodal results\"\"\"
        patterns = []
        
        # Pattern: High cross-modal agreement
        if "cross_modal_insights" in results:
            cross_modal = results["cross_modal_insights"]
            if "entity_correspondences" in cross_modal:
                correspondences = cross_modal["entity_correspondences"]
                if correspondences["total_correspondences"] > 2:
                    patterns.append({
                        "type": "high_cross_modal_agreement",
                        "description": f"Found {correspondences['total_correspondences']} entities appearing across multiple modalities",
                        "strength": "high",
                        "entities": correspondences["cross_modal_entities"][:5]
                    })
        
        # Pattern: Temporal clustering
        if "temporal_alignment" in results:
            temporal = results["temporal_alignment"]
            if "key_moments" in temporal:
                key_moments = temporal["key_moments"]
                multi_modal_moments = [m for m in key_moments if m.get("reason") == "multi_modal_convergence"]
                if multi_modal_moments:
                    patterns.append({
                        "type": "temporal_convergence",
                        "description": f"Found {len(multi_modal_moments)} moments where multiple modalities converge",
                        "strength": "medium",
                        "moments": multi_modal_moments[:3]
                    })
        
        return patterns
    
    def _identify_anomalies(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        # \"\"\"Identify anomalies in multimodal results\"\"\"
       anomalies = []
       
       # Anomaly: Low cross-modal confidence with high individual confidence
       if "fusion_summary" in results:
           fusion_summary = results["fusion_summary"]
           cross_modal_conf = fusion_summary.get("cross_modal_confidence", 0.5)
           
           if cross_modal_conf < 0.3:
               anomalies.append({
                   "type": "cross_modal_inconsistency",
                   "description": "Low agreement between different modalities despite individual high confidence",
                   "severity": "high",
                   "confidence_score": cross_modal_conf
               })
       
       # Anomaly: Sentiment conflicts
       if "cross_modal_insights" in results:
           sentiment_consistency = results["cross_modal_insights"].get("sentiment_consistency", {})
           if sentiment_consistency.get("consistency_score", 1.0) < 0.5:
               anomalies.append({
                   "type": "sentiment_conflict",
                   "description": "Conflicting sentiments detected across modalities",
                   "severity": "medium",
                   "sentiment_distribution": sentiment_consistency.get("sentiment_distribution", {})
               })
       
       # Anomaly: Statistical outliers
       if "statistical_insights" in results:
           stat_anomalies = results["statistical_insights"].get("detected_anomalies", [])
           for anomaly in stat_anomalies:
               anomalies.append({
                   "type": f"statistical_{anomaly['anomaly_type']}",
                   "description": f"Statistical anomaly detected in {anomaly['feature']}",
                   "severity": "low",
                   "details": anomaly
               })
       
       return anomalies
   
    def _identify_trends(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Identify trends in multimodal results\"\"\"
       trends = []
       
       # Trend: Increasing/decreasing patterns
       if "statistical_insights" in results:
           patterns = results["statistical_insights"].get("detected_patterns", [])
           for pattern in patterns:
               if "trend" in pattern["pattern_type"]:
                   trends.append({
                       "type": pattern["pattern_type"],
                       "description": f"Detected {pattern['pattern_type']} in {pattern['feature']}",
                       "strength": pattern["strength"],
                       "feature": pattern["feature"]
                   })
       
       # Trend: Temporal patterns
       if "temporal_alignment" in results:
           temporal = results["temporal_alignment"]
           if "synchronization_score" in temporal:
               sync_score = temporal["synchronization_score"]
               if sync_score > 0.8:
                   trends.append({
                       "type": "high_temporal_synchronization",
                       "description": "Events across modalities are highly synchronized",
                       "strength": sync_score
                   })
               elif sync_score < 0.3:
                   trends.append({
                       "type": "temporal_fragmentation",
                       "description": "Events across modalities are poorly synchronized",
                       "strength": 1.0 - sync_score
                   })
       
       return trends
   
    def _identify_relationships(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Identify relationships in multimodal results\"\"\"
       relationships = []
       
       # Entity relationships
       if "cross_modal_insights" in results:
           entity_correspondences = results["cross_modal_insights"].get("entity_correspondences", {})
           if "correspondences" in entity_correspondences:
               for correspondence in entity_correspondences["correspondences"]:
                   relationships.append({
                       "type": "entity_correspondence",
                       "description": f"Entity '{correspondence['entity_text']}' appears across {len(correspondence['modalities'])} modalities",
                       "strength": correspondence["avg_confidence"],
                       "modalities": correspondence["modalities"]
                   })
       
       # Graph relationships
       if "graph_insights" in results:
           graph = results["graph_insights"]
           if "key_entities" in graph:
               for entity, centrality in graph["key_entities"][:3]:
                   relationships.append({
                       "type": "central_entity",
                       "description": f"Entity '{entity}' has high centrality in the relationship graph",
                       "strength": centrality,
                       "centrality_score": centrality
                   })
       
       return relationships
   
    def _calculate_insight_confidence(self, insights: Dict[str, Any]) -> float:
    #    \"\"\"Calculate overall confidence in generated insights\"\"\"
       confidence_factors = []
       
       # Factor 1: Number of patterns found
       patterns = insights.get("patterns", [])
       if patterns:
           confidence_factors.append(min(len(patterns) / 3.0, 1.0))
       else:
           confidence_factors.append(0.2)
       
       # Factor 2: Anomaly severity
       anomalies = insights.get("anomalies", [])
       high_severity_count = sum(1 for a in anomalies if a.get("severity") == "high")
       if high_severity_count == 0:
           confidence_factors.append(0.9)  # High confidence if no major anomalies
       else:
           confidence_factors.append(max(0.1, 0.9 - high_severity_count * 0.2))
       
       # Factor 3: Relationship strength
       relationships = insights.get("relationships", [])
       if relationships:
           avg_strength = np.mean([r.get("strength", 0.5) for r in relationships])
           confidence_factors.append(avg_strength)
       else:
           confidence_factors.append(0.3)
       
       return float(np.mean(confidence_factors))
   
    async def align_semantically(
       self,
       text_content: str,
       image_descriptions: List[str],
       video_transcripts: List[str]
   ) -> Dict[str, Any]:
    #    \"\"\"Align content semantically across modalities\"\"\"
       try:
           # Combine all text content
           all_content = [text_content] + image_descriptions + video_transcripts
           all_content = [content for content in all_content if content.strip()]
           
           if len(all_content) < 2:
               return {"coherence_score": 1.0, "alignment_matrix": []}
           
           # Generate embeddings
           embeddings = self.models['text_encoder'].encode(all_content)
           
           # Calculate similarity matrix
           similarity_matrix = cosine_similarity(embeddings)
           
           # Calculate coherence score
           coherence_score = float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
           
           # Find best alignments
           alignments = []
           for i in range(len(all_content)):
               for j in range(i + 1, len(all_content)):
                   similarity = similarity_matrix[i][j]
                   if similarity > 0.5:  # Threshold for meaningful alignment
                       alignments.append({
                           "content1_index": i,
                           "content2_index": j,
                           "content1": all_content[i][:100] + "..." if len(all_content[i]) > 100 else all_content[i],
                           "content2": all_content[j][:100] + "..." if len(all_content[j]) > 100 else all_content[j],
                           "similarity": float(similarity)
                       })
           
           # Sort alignments by similarity
           alignments.sort(key=lambda x: x["similarity"], reverse=True)
           
           return {
               "coherence_score": coherence_score,
               "alignment_matrix": similarity_matrix.tolist(),
               "top_alignments": alignments[:5],
               "content_count": len(all_content)
           }
           
       except Exception as e:
           logger.error(f"Semantic alignment error: {str(e)}")
           raise
   
    async def synchronize_temporal(
       self,
       video_timeline: List[Dict[str, Any]],
       audio_events: List[Dict[str, Any]],
       text_timestamps: List[Dict[str, Any]]
   ) -> Dict[str, Any]:
    #    \"\"\"Synchronize events across temporal modalities\"\"\"
       try:
           # Combine all temporal events
           all_events = []
           
           for event in video_timeline:
               all_events.append({
                   "start_time": event.get("timestamp", event.get("start_time", 0)),
                   "end_time": event.get("end_time", event.get("timestamp", 0) + 1),
                   "content": event.get("content", str(event)),
                   "modality": "video",
                   "confidence": event.get("confidence", 0.7)
               })
           
           for event in audio_events:
               all_events.append({
                   "start_time": event.get("start_time", event.get("timestamp", 0)),
                   "end_time": event.get("end_time", event.get("start_time", 0) + 1),
                   "content": event.get("content", str(event)),
                   "modality": "audio",
                   "confidence": event.get("confidence", 0.7)
               })
           
           for event in text_timestamps:
               all_events.append({
                   "start_time": event.get("start_time", event.get("timestamp", 0)),
                   "end_time": event.get("end_time", event.get("start_time", 0) + 1),
                   "content": event.get("content", event.get("text", str(event))),
                   "modality": "text",
                   "confidence": event.get("confidence", 0.7)
               })
           
           # Sort by start time
           all_events.sort(key=lambda x: x["start_time"])
           
           # Calculate synchronization quality
           quality_score = self._calculate_synchronization_score(all_events)
           
           # Detect conflicts (overlapping events with different content)
           conflicts = self._detect_temporal_conflicts(all_events)
           
           # Create synchronized timeline
           synchronized_timeline = self._create_synchronized_timeline(all_events)
           
           return {
               "timeline": synchronized_timeline,
               "quality_score": quality_score,
               "conflicts": conflicts,
               "total_events": len(all_events),
               "modalities": list(set(event["modality"] for event in all_events))
           }
           
       except Exception as e:
           logger.error(f"Temporal synchronization error: {str(e)}")
           raise
   
    def _detect_temporal_conflicts(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #    \"\"\"Detect conflicts in temporal events\"\"\"
       conflicts = []
       
       for i, event1 in enumerate(events):
           for j, event2 in enumerate(events[i+1:], i+1):
               # Check for temporal overlap
               if (event1["start_time"] < event2["end_time"] and 
                   event2["start_time"] < event1["end_time"]):
                   
                   # Check if content is conflicting (different modalities saying different things)
                   if event1["modality"] != event2["modality"]:
                       # Simple conflict detection based on content similarity
                       content_similarity = self._calculate_string_similarity(
                           event1["content"], event2["content"]
                       )
                       
                       if content_similarity < 0.3:  # Low similarity indicates potential conflict
                           conflicts.append({
                               "event1_index": i,
                               "event2_index": j,
                               "overlap_start": max(event1["start_time"], event2["start_time"]),
                               "overlap_end": min(event1["end_time"], event2["end_time"]),
                               "modalities": [event1["modality"], event2["modality"]],
                               "content_similarity": content_similarity,
                               "severity": "high" if content_similarity < 0.1 else "medium"
                           })
       
       return conflicts
   
    def _create_synchronized_timeline(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    #    \"\"\"Create a synchronized timeline from all events\"\"\"
       timeline = []
       
       # Group events by time windows
       time_windows = {}
       window_size = 2.0  # 2-second windows
       
       for event in events:
           window_start = int(event["start_time"] // window_size) * window_size
           if window_start not in time_windows:
               time_windows[window_start] = []
           time_windows[window_start].append(event)
       
       # Create timeline entries
       for window_start in sorted(time_windows.keys()):
           window_events = time_windows[window_start]
           
           # Combine events in the same window
           combined_content = []
           modalities = set()
           avg_confidence = 0
           
           for event in window_events:
               combined_content.append(f"[{event['modality']}] {event['content']}")
               modalities.add(event['modality'])
               avg_confidence += event['confidence']
           
           avg_confidence /= len(window_events)
           
           timeline.append({
               "start_time": window_start,
               "end_time": window_start + window_size,
               "modalities": list(modalities),
               "event_count": len(window_events),
               "combined_content": " | ".join(combined_content),
               "confidence": avg_confidence
           })
       
       return timeline
   
    async def discover_patterns(
       self,
       multimodal_data: Dict[str, Any],
       pattern_types: List[str] = ["recurring", "sequential", "hierarchical"]
   ) -> Dict[str, Any]:
    #    \"\"\"Discover patterns across multimodal data\"\"\"
       try:
           patterns = {
               "recurring": [],
               "sequential": [],
               "hierarchical": [],
               "confidence_scores": {},
               "significance_scores": {}
           }
           
           for pattern_type in pattern_types:
               if pattern_type == "recurring":
                   patterns["recurring"] = self._find_recurring_patterns(multimodal_data)
               elif pattern_type == "sequential":
                   patterns["sequential"] = self._find_sequential_patterns(multimodal_data)
               elif pattern_type == "hierarchical":
                   patterns["hierarchical"] = self._find_hierarchical_patterns(multimodal_data)
           
           # Calculate confidence and significance scores
           for pattern_type, pattern_list in patterns.items():
               if isinstance(pattern_list, list) and pattern_list:
                   patterns["confidence_scores"][pattern_type] = np.mean([p.get("confidence", 0.5) for p in pattern_list])
                   patterns["significance_scores"][pattern_type] = len(pattern_list) / 10.0  # Normalize by expected count
           
           return patterns
           
       except Exception as e:
           logger.error(f"Pattern discovery error: {str(e)}")
           raise
   
    def _find_recurring_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Find recurring patterns in the data\"\"\"
       patterns = []
       
       # Look for recurring entities
       if "cross_modal_insights" in data:
           entity_correspondences = data["cross_modal_insights"].get("entity_correspondences", {})
           if "correspondences" in entity_correspondences:
               for correspondence in entity_correspondences["correspondences"]:
                   if correspondence["occurrences"] > 2:
                       patterns.append({
                           "type": "recurring_entity",
                           "entity": correspondence["entity_text"],
                           "frequency": correspondence["occurrences"],
                           "modalities": correspondence["modalities"],
                           "confidence": correspondence["avg_confidence"]
                       })
       
       # Look for recurring temporal patterns
       if "temporal_alignment" in data:
           timeline = data["temporal_alignment"].get("unified_timeline", [])
           if timeline:
               # Simple frequency analysis of content
               content_frequency = Counter()
               for event in timeline:
                   words = re.findall(r'\b\w+\b', event.get("content", "").lower())
                   content_frequency.update(words)
               
               for word, freq in content_frequency.most_common(5):
                   if freq > 2 and len(word) > 3:  # Filter short words and low frequency
                       patterns.append({
                           "type": "recurring_content",
                           "content": word,
                           "frequency": freq,
                           "confidence": min(freq / 10.0, 1.0)
                       })
       
       return patterns
   
    def _find_sequential_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Find sequential patterns in the data\"\"\"
       patterns = []
       
       # Look for sequential patterns in temporal data
       if "temporal_alignment" in data:
           timeline = data["temporal_alignment"].get("unified_timeline", [])
           if len(timeline) > 2:
               # Look for sequences of events
               for i in range(len(timeline) - 1):
                   current_event = timeline[i]
                   next_event = timeline[i + 1]
                   
                   # Check if events are closely spaced
                   time_gap = next_event["start_time"] - current_event["end_time"]
                   if time_gap < 5.0:  # Events within 5 seconds
                       patterns.append({
                           "type": "sequential_events",
                           "event1": current_event.get("content", "")[:50],
                           "event2": next_event.get("content", "")[:50],
                           "time_gap": time_gap,
                           "modalities": [current_event.get("modality", ""), next_event.get("modality", "")],
                           "confidence": 0.7
                       })
       
       return patterns[:5]  # Limit to top 5 sequential patterns
   
    def _find_hierarchical_patterns(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Find hierarchical patterns in the data\"\"\"
       patterns = []
       
       # Look for hierarchical relationships in graph data
       if "graph_insights" in data:
           graph_data = data["graph_insights"]
           if "centrality_scores" in graph_data:
               centrality_scores = graph_data["centrality_scores"]
               
               # Find nodes with high centrality (likely parent nodes in hierarchy)
               high_centrality_nodes = [
                   (node, score) for node, score in centrality_scores.items() 
                   if score > np.mean(list(centrality_scores.values()))
               ]
               
               for node, score in high_centrality_nodes[:3]:
                   patterns.append({
                       "type": "hierarchical_entity",
                       "parent_entity": node,
                       "centrality_score": score,
                       "description": f"Entity '{node}' has high centrality, suggesting hierarchical importance",
                       "confidence": score
                   })
       
       return patterns
   
    async def detect_anomalies(
       self,
       multimodal_data: Dict[str, Any],
       anomaly_types: List[str] = ["statistical", "semantic", "temporal"]
   ) -> Dict[str, Any]:
    #    \"\"\"Detect anomalies across multimodal data\"\"\"
       try:
           anomalies = {
               "statistical": [],
               "semantic": [],
               "temporal": [],
               "scores": {},
               "severity": {}
           }
           
           for anomaly_type in anomaly_types:
               if anomaly_type == "statistical":
                   anomalies["statistical"] = self._detect_statistical_anomalies_advanced(multimodal_data)
               elif anomaly_type == "semantic":
                   anomalies["semantic"] = await self._detect_semantic_anomalies(multimodal_data)
               elif anomaly_type == "temporal":
                   anomalies["temporal"] = self._detect_temporal_anomalies(multimodal_data)
           
           # Calculate anomaly scores and severity
           for anomaly_type, anomaly_list in anomalies.items():
               if isinstance(anomaly_list, list) and anomaly_list:
                   scores = [a.get("score", 0.5) for a in anomaly_list]
                   anomalies["scores"][anomaly_type] = float(np.mean(scores)) if scores else 0.0
                   
                   # Determine severity based on score and count
                   avg_score = np.mean(scores) if scores else 0.0
                   severity = "low"
                   if avg_score > 0.7 and len(anomaly_list) > 2:
                       severity = "high"
                   elif avg_score > 0.5 or len(anomaly_list) > 1:
                       severity = "medium"
                   
                   anomalies["severity"][anomaly_type] = severity
           
           return anomalies
           
       except Exception as e:
           logger.error(f"Anomaly detection error: {str(e)}")
           raise
   
    def _detect_statistical_anomalies_advanced(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Detect statistical anomalies with advanced methods\"\"\"
       anomalies = []
       
       # Extract numerical features
       features = {}
       
       # From fusion summary
       if "fusion_summary" in data:
           fusion_summary = data["fusion_summary"]
           for key, value in fusion_summary.items():
               if isinstance(value, (int, float)):
                   features[f"fusion_{key}"] = [value]
       
       # From statistical insights
       if "statistical_insights" in data:
           stat_insights = data["statistical_insights"]
           if "feature_statistics" in stat_insights:
               for feature_name, stats in stat_insights["feature_statistics"].items():
                   features[feature_name] = [stats.get("mean", 0)]
       
       # Detect outliers in each feature
       for feature_name, values in features.items():
           if len(values) >= 1:
               value = values[0]
               
               # Use domain knowledge for anomaly detection
               if "confidence" in feature_name.lower():
                   if value < 0.2:
                       anomalies.append({
                           "type": "low_confidence",
                           "feature": feature_name,
                           "value": value,
                           "score": 1.0 - value,  # Higher score for lower confidence
                           "description": f"Unusually low confidence detected in {feature_name}"
                       })
                   elif value > 0.98:
                       anomalies.append({
                           "type": "suspiciously_high_confidence",
                           "feature": feature_name,
                           "value": value,
                           "score": (value - 0.95) * 20,  # Scale to 0-1
                           "description": f"Suspiciously high confidence detected in {feature_name}"
                       })
       
       return anomalies
   
    async def _detect_semantic_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Detect semantic anomalies in multimodal data\"\"\"
       anomalies = []
       
       # Check for semantic inconsistencies
       if "cross_modal_insights" in data:
           cross_modal = data["cross_modal_insights"]
           
           # Check sentiment consistency
           sentiment_consistency = cross_modal.get("sentiment_consistency", {})
           consistency_score = sentiment_consistency.get("consistency_score", 1.0)
           
           if consistency_score < 0.3:
               anomalies.append({
                   "type": "sentiment_inconsistency",
                   "score": 1.0 - consistency_score,
                   "description": "Major sentiment inconsistency detected across modalities",
                   "details": sentiment_consistency
               })
           
           # Check entity correspondence anomalies
           entity_correspondences = cross_modal.get("entity_correspondences", {})
           if entity_correspondences.get("total_correspondences", 0) == 0:
               anomalies.append({
                   "type": "no_entity_overlap",
                   "score": 0.8,
                   "description": "No common entities found across modalities - possible content mismatch"
               })
       
       return anomalies
   
    def _detect_temporal_anomalies(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    #    \"\"\"Detect temporal anomalies in multimodal data\"\"\"
       anomalies = []
       
       if "temporal_alignment" in data:
           temporal = data["temporal_alignment"]
           
           # Check synchronization quality
           sync_score = temporal.get("synchronization_score", 1.0)
           if sync_score < 0.2:
               anomalies.append({
                   "type": "poor_synchronization",
                   "score": 1.0 - sync_score,
                   "description": "Poor temporal synchronization between modalities",
                   "sync_score": sync_score
               })
           
           # Check for unusual gaps in timeline
           timeline = temporal.get("unified_timeline", [])
           if len(timeline) > 1:
               gaps = []
               for i in range(1, len(timeline)):
                   gap = timeline[i]["start_time"] - timeline[i-1]["end_time"]
                   gaps.append(gap)
               
               if gaps:
                   avg_gap = np.mean(gaps)
                   std_gap = np.std(gaps)
                   
                   # Find unusually large gaps
                   for i, gap in enumerate(gaps):
                       if gap > avg_gap + 2 * std_gap and gap > 10:  # Gap > 10 seconds and 2 std devs
                           anomalies.append({
                               "type": "temporal_gap",
                               "score": min((gap - avg_gap) / (3 * std_gap), 1.0),
                               "description": f"Unusually large temporal gap detected ({gap:.1f} seconds)",
                               "gap_duration": gap,
                               "position": i
                           })
       
       return anomalies
   
    async def synthesize_narrative(
       self,
       insights: Dict[str, Any],
       style: str = "comprehensive",
       audience: str = "general"
   ) -> Dict[str, Any]:
    #    \"\"\"Synthesize a narrative from multimodal insights\"\"\"
       try:
           narrative_parts = []
           key_points = []
           
           # Introduction
           if style == "comprehensive":
               narrative_parts.append("Based on comprehensive multimodal analysis, the following insights have been discovered:")
           elif style == "summary":
               narrative_parts.append("Key findings from multimodal analysis:")
           else:
               narrative_parts.append("Analysis results:")
           
           # Patterns section
           if "patterns" in insights and insights["patterns"]:
               narrative_parts.append("\n\nPatterns Identified:")
               for pattern in insights["patterns"][:3]:  # Top 3 patterns
                   description = pattern.get("description", "Pattern detected")
                   narrative_parts.append(f"• {description}")
                   key_points.append(pattern.get("type", "pattern"))
           
           # Relationships section
           if "relationships" in insights and insights["relationships"]:
               narrative_parts.append("\n\nKey Relationships:")
               for relationship in insights["relationships"][:3]:
                   description = relationship.get("description", "Relationship found")
                   narrative_parts.append(f"• {description}")
                   key_points.append(relationship.get("type", "relationship"))
           
           # Anomalies section
           if "anomalies" in insights and insights["anomalies"]:
               high_severity_anomalies = [a for a in insights["anomalies"] if a.get("severity") == "high"]
               if high_severity_anomalies:
                   narrative_parts.append("\n\nImportant Anomalies:")
                   for anomaly in high_severity_anomalies[:2]:
                       description = anomaly.get("description", "Anomaly detected")
                       narrative_parts.append(f"• {description}")
                       key_points.append(f"anomaly_{anomaly.get('type', 'unknown')}")
           
           # Trends section
           if "trends" in insights and insights["trends"]:
               narrative_parts.append("\n\nObserved Trends:")
               for trend in insights["trends"][:2]:
                   description = trend.get("description", "Trend observed")
                   narrative_parts.append(f"• {description}")
                   key_points.append(trend.get("type", "trend"))
           
           # Conclusion
           confidence = insights.get("overall_confidence", 0.7)
           if confidence > 0.8:
               narrative_parts.append(f"\n\nThe analysis shows high confidence ({confidence:.2f}) in these findings.")
           elif confidence > 0.6:
               narrative_parts.append(f"\n\nThe analysis shows moderate confidence ({confidence:.2f}) in these findings.")
           else:
               narrative_parts.append(f"\n\nThe analysis shows limited confidence ({confidence:.2f}) in these findings, suggesting further investigation may be needed.")
           
           # Combine narrative
           full_narrative = "".join(narrative_parts)
           
           # Calculate readability (simplified)
           readability_score = self._calculate_readability(full_narrative, audience)
           
           return {
               "text": full_narrative,
               "key_points": key_points,
               "confidence": float(confidence),
               "readability": readability_score,
               "word_count": len(full_narrative.split()),
               "style": style,
               "audience": audience
           }
           
       except Exception as e:
           logger.error(f"Narrative synthesis error: {str(e)}")
           raise
   
    def _calculate_readability(self, text: str, audience: str) -> float:
    #    \"\"\"Calculate readability score based on text complexity and audience\"\"\"
       # Simplified readability calculation
       words = text.split()
       sentences = text.split('.')
       
       if not words or not sentences:
           return 0.5
       
       avg_words_per_sentence = len(words) / len(sentences)
       
       # Simple readability heuristics
       if audience == "technical":
           # Technical audience can handle more complex text
           readability = min(0.9 - (avg_words_per_sentence - 15) * 0.02, 1.0)
       else:
           # General audience needs simpler text
           readability = min(0.9 - (avg_words_per_sentence - 10) * 0.03, 1.0)
       
       return max(readability, 0.1)  # Minimum readability score
   
    def get_available_algorithms(self) -> List[str]:
    #    \"\"\"Get list of available fusion algorithms\"\"\"
       return list(self.fusion_algorithms.keys())
   
    def get_algorithm_details(self) -> Dict[str, Dict[str, str]]:
    #    \"\"\"Get detailed information about available algorithms\"\"\"
       return {
           "semantic": {
               "description": "Fuses content based on semantic similarity and meaning",
               "input": "Text content from all modalities",
               "output": "Semantic clusters, topics, and alignment scores"
           },
           "temporal": {
               "description": "Aligns and synchronizes time-based events across modalities",
               "input": "Timestamped events from video, audio, and text",
               "output": "Unified timeline with synchronization quality metrics"
           },
           "cross_modal": {
               "description": "Finds relationships and correspondences between different modalities",
               "input": "Analysis results from text, image, and video processing",
               "output": "Entity correspondences, sentiment consistency, relationship graphs"
           },
           "statistical": {
               "description": "Applies statistical methods to identify patterns and anomalies",
               "input": "Numerical features extracted from all modalities",
               "output": "Statistical patterns, anomalies, and feature correlations"
           },
           "graph_based": {
               "description": "Uses graph theory to model relationships between multimodal elements",
               "input": "Entities and relationships from all modalities",
               "output": "Network analysis results, centrality scores, communities"
           }
       }
   
    def get_memory_usage(self) -> Dict[str, float]:
    #    \"\"\"Get current memory usage\"\"\"
       process = psutil.Process()
       memory_info = process.memory_info()
       
       return {
           "rss_mb": memory_info.rss / 1024 / 1024,
           "vms_mb": memory_info.vms / 1024 / 1024,  
           "cpu_percent": process.cpu_percent()
       }
   
    def get_model_info(self) -> Dict[str, Any]:
    #    \"\"\"Get information about loaded models\"\"\"
       return {
           "models_loaded": self.models_loaded,
           "device": self.device,
           "available_algorithms": self.get_available_algorithms(),
           "loaded_models": list(self.models.keys()) if self.models_loaded else [],
           "memory_usage": self.get_memory_usage(),
           "capabilities": [
               "semantic_fusion", "temporal_alignment", "cross_modal_correlation",
               "insight_generation", "pattern_discovery", "anomaly_detection",
               "relationship_mapping", "narrative_synthesis"
           ]
       }