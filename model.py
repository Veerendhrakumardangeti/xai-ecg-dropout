import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, resnet18, mobilenet_v2


class ModelEfficientNetV2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = efficientnet_v2_s(weights=None)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        self.ecg_features = backbone.features
        self.ecg_pool = backbone.avgpool
        ecg_out_dim = backbone.classifier[-1].in_features

        self.classifier = nn.Sequential(
            nn.Linear(ecg_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_meta, y_ecg):
        y = y_ecg.permute(0, 3, 1, 2)
        y = F.pad(y, (0, 0, 10, 10))
        y = self.ecg_features(y)
        y = self.ecg_pool(y).flatten(1)
        return self.classifier(y)



class ModelResNet18(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        in_features = backbone.fc.in_features  # 512
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_meta, y_ecg):
        y = y_ecg.permute(0, 3, 1, 2)
        y = F.pad(y, (0, 0, 10, 10))
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        y = self.layer4(y)
        y = self.avgpool(y).flatten(1)
        return self.classifier(y)



class ModelMobileNetV2(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        backbone = mobilenet_v2(weights=None)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        self.features = backbone.features

        in_features = backbone.classifier[1].in_features  # 1280
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_meta, y_ecg):
        y = y_ecg.permute(0, 3, 1, 2)
        y = F.pad(y, (0, 0, 10, 10))
        y = self.features(y)
        y = nn.functional.adaptive_avg_pool2d(y, (1, 1)).flatten(1)
        return self.classifier(y)


def build_model(model_name, num_classes=5):
    """simple factory to pick a model by name string"""
    models = {
        "efficientnet": ModelEfficientNetV2,
        "resnet18": ModelResNet18,
        "mobilenetv2": ModelMobileNetV2,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(models.keys())}")
    return models[model_name](num_classes=num_classes)
