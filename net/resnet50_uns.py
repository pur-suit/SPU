import torch
import torch.nn as nn
import torch.nn.functional as F
from net import resnet50

class _ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling (ASPP)
    """

    def __init__(self, in_ch, out_ch, rates):
        super(_ASPP, self).__init__()
        for i, rate in enumerate(rates):
            self.add_module(
                "c{}".format(i),
                nn.Conv2d(in_ch, out_ch, 3, 1, padding=rate, dilation=rate, bias=True),
            )

        for m in self.children():
            nn.init.normal_(m.weight, mean=0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return sum([stage(x) for stage in self.children()])


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.resnet50 = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.stage1 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool,
                                    self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        astrous_rates = [6, 12, 18, 24]

        self.label_enc1 = nn.Linear(20, 1024+2048)

        self.classifier1 = nn.Sequential(
            nn.Dropout(0.1),
            _ASPP(in_ch=1024+2048, out_ch=21, rates=astrous_rates)
        )

        self.label_enc2 = nn.Linear(20, 1024)

        self.classifier2 = nn.Sequential(
            nn.Dropout(0.1),
            _ASPP(in_ch=1024, out_ch=21, rates=astrous_rates)
        )

        self.backbone = nn.ModuleList([self.stage1, self.stage2, self.stage3, self.stage4])
        self.newly_added = nn.ModuleList([self.classifier1, self.classifier2, self.label_enc1, self.label_enc2])


    def forward(self, img, label_cls):

        y2 = self.label_enc2(label_cls).unsqueeze(-1).unsqueeze(-1)

        y1 = self.label_enc1(label_cls).unsqueeze(-1).unsqueeze(-1)

        x = self.stage1(img)

        x = self.stage2(x)

        x = self.stage3(x)

        x1 = x * y2
        x1 = self.classifier2(x1)

        x2 = torch.cat((self.stage4(x), x), 1)
        x2 = x2 * y1
        x2 = self.classifier1(x2)

        return x1, x2

    def train(self, mode=True):
        for p in self.resnet50.conv1.parameters():
            p.requires_grad = False
        for p in self.resnet50.bn1.parameters():
            p.requires_grad = False

    def trainable_parameters(self):

        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))


class CAM(Net):

    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, x, y):

        y2 = self.label_enc2(y).unsqueeze(-1).unsqueeze(-1)
        y1 = self.label_enc1(y).unsqueeze(-1).unsqueeze(-1)


        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)


        x1 = x * y2
        x1 = self.classifier2(x1)

        x2 = torch.cat((self.stage4(x), x), 1)
        x2 = x2 * y1
        x2 = self.classifier1(x2)

        logit = (x2[0] + x2[1].flip(-1)) / 2

        return logit
