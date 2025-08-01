import torch.nn as nn
from models.utils_prompt_pool import PromptPool


class PromptEnhancedModel(nn.Module):
    def __init__(self, backbone, projector, prompt_pool=None, top_k=5):
        super().__init__()
        self.backbone = backbone
        self.projector = projector
        self.prompt_pool = prompt_pool
        self.top_k = top_k

    def forward(self, x):
        # Extract features from backbone
        features = self.backbone(x)

        # Enhance features using prompt pool if available
        if self.prompt_pool is not None:
            features = self.prompt_pool.enhance_features(features, top_k=self.top_k)

        # Pass through projector
        proj_features, logits = self.projector(features)

        return proj_features, logits