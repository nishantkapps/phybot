import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class PoseTransformerLSTM(nn.Module):

    def __init__(self, num_joints, num_classes, d_model=128):
        super().__init__()
        self.embedding = nn.Linear(num_joints * 2, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.lstm = nn.LSTM(d_model, 256, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, joints):
        B, T, J, _ = joints.shape
        x = joints.view(B, T, J*2)
        x = self.embedding(x)
        x = self.transformer(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
    
# Example usage:
if __name__ == '__main__':
    # Define parameters
    BATCH_SIZE = 4
    SEQUENCE_LENGTH = 10  # Number of frames in a sequence
    NUM_JOINTS = 17       # e.g., for COCO dataset
    NUM_CLASSES = 5       # e.g., 5 different activities
    
    # Generate some dummy data
    dummy_input = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, NUM_JOINTS, 2)
    
    # Create the model
    model = PoseTransformerLSTM(num_joints=NUM_JOINTS, num_classes=NUM_CLASSES)
    
    # Get model output
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape (logits): {output.shape}")
