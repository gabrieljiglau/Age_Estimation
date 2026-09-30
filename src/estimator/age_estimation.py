import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models import ResNet50_Weights
from omegaconf import DictConfig

class AgeEstimator(nn.Module):

    """
        age estimator using the pretrained ResNet50 on ImageNet
    """

    def __init__(
            self,
            num_classes: int=80,
            dropout_prob: float=0.3
    ) -> None:

        super(AgeEstimator, self).__init__()

        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.estimator.backbone = backbone
        self._make_head(self.num_classes, self.dropout_prob)


    def _make_head(
            self,
            num_classes: int=80,
            dropout_prob: float=0.3
    ) -> None:

        """
            freezes all the layers (except the last one) of the pre-trained network
            and adds a new fully connected layer at the end
        """

        num_features = self.estimator.fc.in_features

        self.estimator.head = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(num_features, num_classes)
        )


    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:

        """

        :param x: data of shape [batch_size, 3 channels, height, width]
        :return: the vector of raw (non-normalized) predictions the model generated,
        """

        return self.estimator(x)

    @property
    def get_num_parameters(self) -> float:
        """
        total trainable parameters (in millions), rounded to 2 decimals
        """
        return round(sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6, 2)


def build_model(cfg: DictConfig) -> AgeEstimator:
    """
    instantiate the AgeEstimator, based on an OmegaConf config file
    """

    return AgeEstimator(
        num_classes=cfg.num_classes,
        dropout_prob=cfg.model.dropout)