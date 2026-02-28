"""
Optimized Real-Time Sign Recognition
Designed for minimum latency and maximum performance
"""

import torch
import numpy as np
import json
import os
import cv2
from collections import deque
import time

class RealtimeSignRecognizer:
    """
    Optimized sign recognizer for real-time performance
    """
    
    def __init__(self, model_path, metadata_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.idx_to_label = {int(k): v for k, v in metadata['idx_to_label'].items()}
        self.num_classes = metadata['num_classes']
        self.max_frames = metadata.get('max_frames', 64)
        self.landmark_dim = metadata.get('landmark_dim', 1662)
        
        # Load model
        print(f"Loading model for real-time recognition...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Import model
        from ..models.lstm_model import EnhancedLSTMModel
        self.model = EnhancedLSTMModel(
            input_dim=self.landmark_dim,
            num_classes=self.num_classes
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Optimize model for inference
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Use FP16 for faster inference
        
        # MediaPipe extractor
        from ..models.mediapipe_extractor import MediaPipeExtractor
        self.extractor = MediaPipeExtractor()
        
        # Frame buffer for temporal smoothing
        self.frame_buffer = deque(maxlen=self.max_frames)
        self.prediction_buffer = deque(maxlen=5)  # Smooth predictions
        
        # Performance tracking
        self.frame_count = 0
        self.total_time = 0
        
        print(f"✓ Real-time recognizer ready on {self.device}")
    
    def process_frame(self, frame):
        """
        Process a single frame and return prediction
        Optimized for speed
        """
        start_time = time.time()
        
        # Extract features from frame
        features = self.extractor.extract_frame(frame)
        
        # Add to buffer
        self.frame_buffer.append(features)
        
        # Need minimum frames for prediction
        if len(self.frame_buffer) < 16:  # Minimum 16 frames
            return None
        
        # Prepare sequence (use last N frames)
        sequence = list(self.frame_buffer)
        
        # Pad if needed
        while len(sequence) < self.max_frames:
            sequence.append(np.zeros(self.landmark_dim))
        
        # Convert to tensor
        features_tensor = torch.from_numpy(np.array(sequence[:self.max_frames])).float()
        features_tensor = features_tensor.unsqueeze(0).to(self.device)
        
        if self.device.type == 'cuda':
            features_tensor = features_tensor.half()
        
        # Predict
        with torch.no_grad():
            logits, _ = self.model(features_tensor)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_label = self.idx_to_label[predicted.item()]
        confidence_value = confidence.item()
        
        # Add to prediction buffer for smoothing
        self.prediction_buffer.append((predicted_label, confidence_value))
        
        # Get most common prediction from buffer
        if len(self.prediction_buffer) >= 3:
            labels = [p[0] for p in self.prediction_buffer]
            confidences = [p[1] for p in self.prediction_buffer]
            
            # Most common label
            from collections import Counter
            label_counts = Counter(labels)
            smoothed_label = label_counts.most_common(1)[0][0]
            smoothed_confidence = np.mean([c for l, c in self.prediction_buffer if l == smoothed_label])
        else:
            smoothed_label = predicted_label
            smoothed_confidence = confidence_value
        
        # Track performance
        elapsed = time.time() - start_time
        self.frame_count += 1
        self.total_time += elapsed
        
        fps = 1.0 / elapsed if elapsed > 0 else 0
        avg_fps = self.frame_count / self.total_time if self.total_time > 0 else 0
        
        return {
            'label': smoothed_label,
            'confidence': float(smoothed_confidence),
            'raw_label': predicted_label,
            'raw_confidence': float(confidence_value),
            'fps': fps,
            'avg_fps': avg_fps,
            'latency_ms': elapsed * 1000,
            'buffer_size': len(self.frame_buffer)
        }
    
    def reset_buffer(self):
        """Reset frame buffer (call when starting new sign)"""
        self.frame_buffer.clear()
        self.prediction_buffer.clear()
    
    def get_stats(self):
        """Get performance statistics"""
        return {
            'total_frames': self.frame_count,
            'total_time': self.total_time,
            'avg_fps': self.frame_count / self.total_time if self.total_time > 0 else 0,
            'avg_latency_ms': (self.total_time / self.frame_count * 1000) if self.frame_count > 0 else 0
        }
    
    def close(self):
        """Cleanup resources"""
        self.extractor.close()
