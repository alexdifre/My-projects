import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from CONTR_DISTILL.models.resnetv2 import ResNet18, ResNet , BasicBlock


class TeachResNet18(ResNet):  
    def __init__(self, n_cls, **kwargs):
        super().__init__(block= BasicBlock, num_blocks=[2, 2, 2, 2], num_classes = n_cls, **kwargs)

        
        # load pre-trained weights from ResNet18 torchvision
        pretrained_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained_model.state_dict()
        teach_dict = self.state_dict()

        # filter pre-trained weights, excluding 'fc' layer ones
        pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in teach_dict and k != 'fc.weight' and k != 'fc.bias' and v.shape == teach_dict[k].shape}

        # update state_dict of the model
        teach_dict.update(pretrained_dict_filtered)
        self.load_state_dict(teach_dict, strict=False)
        
      