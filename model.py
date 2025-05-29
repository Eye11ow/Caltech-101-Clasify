import torchvision.models as models
import torch.nn as nn

def create_model(pretrained=True, num_classes=101):
    model = models.resnet18(pretrained=pretrained)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model



