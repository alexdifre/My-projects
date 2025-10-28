import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from CONTR_DISTILL.models.resnetv2 import ResNet50 ,ResNet, Bottleneck


class TeachResNet50(ResNet):  # Eredita da nn.Module, non da ResNet
    def __init__(self, n_cls, **kwargs):
        super().__init__(block= Bottleneck, num_blocks=[3, 4, 6, 3], num_classes = n_cls, **kwargs)

        # load pre-trained weights from ResNet34 torchvision
        pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        teach_dict = self.state_dict()

        # filter pre-trained weights, excluding 'fc' layer ones
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in teach_dict and k != 'fc.weight' and k != 'fc.bias' and v.shape == teach_dict[k].shape}

        # update state_dict of the model
        teach_dict.update(pretrained_dict_filtered)
        self.load_state_dict(teach_dict, strict=False)
        