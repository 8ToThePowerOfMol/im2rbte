import copy

import torch
import torch.nn as nn
import torchvision

from model import initialize_model


# Networks #


class Image2ImageTranslation(nn.Module):
    def __init__(self, params):
        super(Image2ImageTranslation, self).__init__()
        path = params.pop('path')
        self.frozen = params.pop('frozen')
        assert self.frozen, "Only frozen model is currently supported."
        self.model = initialize_model(params)
        self.model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        if self.frozen:
            self.model.eval()

    def train(self, mode=True):
        if self.frozen:
            super().train(False)

    def parameters(self, recurse=True):
        if self.frozen:
            return []
        return self.model.parameters(recurse)

    def forward(self, images):
        with torch.no_grad():  # Assuming, the model is frozen
            if isinstance(images, list):
                return [self(image) for image in images]

            if self.use_grayscale:  # shape [B, 3, H, W] to shape [B, 1, H, W]
                images = rgb_to_grayscale(images)

            if len(images.shape) == 3:  # shape [C, H, W]
                images = self.model(images.unsqueeze(0)).squeeze(0)
            elif len(images.shape) == 4:  # shape [B, C, H, W]
                if images.size(1) > 1:  # shape [B, C>1, H, W]
                    images = torch.chunk(images, chunks=images.size(1), dim=1)
                    images = [self.model(x) for x in images]
                    images = torch.cat(images, dim=1)
                else:  # shape [B, 1, H, W]
                    images = self.model(images)
            else:
                raise RuntimeError(f"Input images have shape {images.shape}, but it should be 3 or 4 long")

            return images


class full_network(torch.nn.Module):
    def __init__(self, backbone):
        super(full_network, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        x = self.backbone(x)

        return x


class SequentialNetwork(nn.Module):
    def __init__(self, sequence):
        super(SequentialNetwork, self).__init__()
        self.sequence = torch.nn.Sequential(*sequence)

    def forward(self, x, **kwargs):
        return self.sequence(x, **kwargs)


# Initialization #


NETWORK_LABELS = {
    "image2image": Image2ImageTranslation,
}

def initizalize_network(architecture, no_classes, load_model=None, pretrained=True, sequence=None):
    backbone = initialize_backbone(architecture=architecture, no_classes=no_classes,
                                   pretrained=pretrained)
    if load_model == 'None':
        load_model = None
    if load_model is not None:
        state = torch.load(load_model, map_location='cpu')
        backbone_state_dict = {(key.split('.', 1)[1] if key.startswith('backbone.') else key):
                               value for key, value in state['state_dict'].items()}
        try:
            backbone.load_state_dict(backbone_state_dict)
        except RuntimeError:
            model_dict = backbone.state_dict()
            backbone_state_dict = {key: value for key, value in backbone_state_dict.items()
                                   if ('fc' not in key and 'classifier' not in key)}
            model_dict.update(backbone_state_dict)
            backbone.load_state_dict(model_dict)
            not_loaded = set(model_dict.keys()) - set(backbone_state_dict.keys())
            for keys in not_loaded:
                print('Not loaded in the backbone:', keys)

    if not sequence:
        return full_network(backbone=backbone)

    return SequentialNetwork(
        [NETWORK_LABELS[x.pop("name")](x) for x in copy.deepcopy(sequence)] + [full_network(backbone=backbone)]
    )


def initialize_backbone(architecture, no_classes, pretrained):
    if architecture.lower() == 'resnet18':
        net = torchvision.models.resnet18(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(512, no_classes)
    elif architecture.lower() == 'resnet50':
        net = torchvision.models.resnet50(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(2048, no_classes)
    elif architecture.lower() == 'resnet101':
        net = torchvision.models.resnet101(pretrained=pretrained)
        if no_classes != 1000:
            net.fc = torch.nn.Linear(2048, no_classes)

    return net
