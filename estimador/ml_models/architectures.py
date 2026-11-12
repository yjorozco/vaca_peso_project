import torch
import torch.nn as nn
import torchvision.models as models

class FiLMLayer(nn.Module):
    def __init__(self, in_features: int, cond_dim: int = 1):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(cond_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(16, in_features * 2)
        )
        nn.init.zeros_(self.generator[-1].weight)
        nn.init.zeros_(self.generator[-1].bias)

    def forward(self, img_feats: torch.Tensor, dist_scalar: torch.Tensor) -> torch.Tensor:
        film_params = self.generator(dist_scalar)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        return img_feats * (1.0 + gamma) + beta

class CowWeightRegressorFiLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.resnet18(weights=None)
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.film = FiLMLayer(in_features=num_ftrs, cond_dim=1)
        self.regressor = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x_img: torch.Tensor, dist_scalar: torch.Tensor) -> torch.Tensor:
        if dist_scalar.ndim == 1:
            dist_scalar = dist_scalar.unsqueeze(1)
        img_feats = self.base_model(x_img)
        modulated_feats = self.film(img_feats, dist_scalar)
        return self.regressor(modulated_feats)

class CowPostureClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(CowPostureClassifier, self).__init__()
        self.base_model = models.resnet50(weights=None)
        original_conv1 = self.base_model.conv1
        self.base_model.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x_img: torch.Tensor) -> torch.Tensor:
        img_feats = self.base_model(x_img)
        return self.classifier(img_feats)
