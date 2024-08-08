import torch
import torch.nn as nn
from vit import ViT

class CBViT(nn.Module):
    def __init__(self):
        super(CBViT, self).__init__()
        self.ViT = ViT(pool='mean', image_size=28, patch_size=4, num_classes=10, channels=2, dim=64, depth=6, heads=8, mlp_dim=128)
        self.ViT.mlp_head = torch.nn.Identity()
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x, attns = self.ViT(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x, attns
