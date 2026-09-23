"""Torchvision ResNet initialization with a visible, shared weights directory."""

import torch
import torchvision
from torchvision.ops.misc import FrozenBatchNorm2d

from alohamini.paths import WorkspacePaths


def make_resnet(config):
    """Keep torchvision initialization semantics; change only the cache directory."""
    weights = torchvision.models.get_model_weights(config.vision_backbone).verify(
        config.pretrained_backbone_weights
    )
    backbone = getattr(torchvision.models, config.vision_backbone)(
        replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
        weights=None,
        norm_layer=FrozenBatchNorm2d,
    )
    if weights is not None:
        state = torch.hub.load_state_dict_from_url(
            weights.url,
            model_dir=str(WorkspacePaths().pretrained),
            map_location="cpu",
            check_hash=True,
            weights_only=True,
        )
        backbone.load_state_dict(state)
    return backbone
