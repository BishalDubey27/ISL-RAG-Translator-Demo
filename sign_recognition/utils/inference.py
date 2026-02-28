"""
Sign Recognition Inference Module
"""

import torch
import numpy as np
import json
import os
from ..models.mediapipe_extractor import MediaPipeExtractor
from ..models.lstm_model import EnhancedLSTMModel
from ..models.transformer_model import TransformerModel  # Import INCLUDE model

class SignRecognitionInference:
    def __init__(self, model_path, metadata_path, device='cuda', model_type='lstm'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type  # Add model type to support multiple models

        # Load metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.idx_to_label = {int(k): v for k, v in metadata['idx_to_label'].items()}
        self.num_classes = metadata['num_classes']
        self.max_frames = metadata.get('max_frames', 64)
        self.landmark_dim = metadata.get('landmark_dim', 1662)

        # Load model based on type
        print(f"Loading {model_type} model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)

        if model_type == 'lstm':
            self.model = EnhancedLSTMModel(
                input_dim=self.landmark_dim,
                num_classes=self.num_classes
            )
        elif model_type == 'transformer':
            self.model = TransformerModel(
                input_dim=self.landmark_dim,
                num_classes=self.num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # MediaPipe extractor
        self.extractor = MediaPipeExtractor()

        print(f"✓ {model_type.capitalize()} model loaded on {self.device}")
    
    def predict_video(self, video_path):
        """Predict sign from video file"""
        # Extract features
        features = self.extractor.extract_video(video_path, self.max_frames)
        features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            if self.model_type == 'lstm':
                logits, attention_weights = self.model(features_tensor)
            elif self.model_type == 'transformer':
                logits = self.model(features_tensor)  # Transformer may not return attention weights
                attention_weights = None

            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(1)

        predicted_label = self.idx_to_label[predicted.item()]

        # Get top 5 predictions
        top5_probs, top5_idx = torch.topk(probabilities[0], min(5, len(self.idx_to_label)))
        top5_predictions = [
            {'label': self.idx_to_label[idx.item()], 'confidence': prob.item()}
            for idx, prob in zip(top5_idx, top5_probs)
        ]

        return {
            'label': predicted_label,
            'confidence': confidence.item(),
            'top5_predictions': top5_predictions,
            'attention_weights': attention_weights.cpu().numpy() if attention_weights is not None else None
        }
    
    def close(self):
        self.extractor.close()