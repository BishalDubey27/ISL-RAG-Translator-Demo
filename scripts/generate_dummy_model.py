import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import torch
from sign_recognition.models.transformer_model import TransformerModel

def save_dummy_model():
    model = TransformerModel(input_dim=1662, num_classes=3)
    dummy_state_dict = model.state_dict()
    torch.save({'model_state_dict': dummy_state_dict}, 'models/include_model.pth')

if __name__ == "__main__":
    save_dummy_model()