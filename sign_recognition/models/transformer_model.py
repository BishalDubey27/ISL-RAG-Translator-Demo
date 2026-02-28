import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        # Define a simple transformer-based architecture
        self.embedding = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, 128)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, 128)
        x = self.transformer(x)  # (seq_len, batch_size, 128)
        x = x.mean(dim=0)  # (batch_size, 128)
        logits = self.fc(x)  # (batch_size, num_classes)
        return logits