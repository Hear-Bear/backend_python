from torch import nn
import torchvision.models as models


class AudioResNet(nn.Module):
    def __init__(self, num_classes, backbone='resnet18', pretrained=False):
        """
        Args:
            num_classes (int): 분류할 클래스 수  
            backbone (str): backbone 모델 이름
            pretrained (bool): 사전 학습 여부
        """
        super(AudioResNet, self).__init__()

        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        if backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        if backbone == 'resnext50_32x4d':
            self.backbone = models.resnext50_32x4d(pretrained=pretrained)
        if backbone == 'resnext101_32x8d':
            self.backbone = models.resnext101_32x8d(pretrained=pretrained)
        if backbone == 'resnext152_32x8d':
            self.backbone = models.resnet152(pretrained=pretrained)

        # 기본 모델은 3채널 입력을 기대하므로, 1채널 입력에 맞게 첫 conv layer 수정
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.model.conv1.out_channels,
            kernel_size=self.model.conv1.kernel_size,
            stride=self.model.conv1.stride,
            padding=self.model.conv1.padding,
            bias=False
        )
        # 마지막 fc layer를 num_classes에 맞게 수정
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)