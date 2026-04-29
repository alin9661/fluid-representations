"""Per-frame 2D ViT encoder for multi-channel physics frames."""

import torch
from torch import nn

import stable_pretraining as spt


def build_per_frame_encoder(
    in_chans: int = 11,
    image_size: int = 256,
    patch_size: int = 16,
    scale: str = "tiny",
    pretrained: bool = False,
):
    """Wraps `spt.backbone.utils.vit_hf` so it accepts arbitrary input channels.

    HF `ViTConfig` supports `num_channels`; passing it through kwargs makes
    `ViTPatchEmbeddings` instantiate `Conv2d(num_channels, hidden, k, s)` directly,
    no projection swap needed.

    Returns the bare HF `ViTModel`. Caller is expected to read
    `output.last_hidden_state[:, 0]` for the CLS embedding.
    """
    encoder = spt.backbone.utils.vit_hf(
        scale,
        patch_size=patch_size,
        image_size=image_size,
        pretrained=pretrained,
        use_mask_token=False,
        num_channels=in_chans,
    )
    if encoder.config.num_channels != in_chans:
        # Defensive fallback: some HF versions ignore num_channels for non-3 inputs.
        proj = encoder.embeddings.patch_embeddings.projection
        new_proj = nn.Conv2d(
            in_chans,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            bias=proj.bias is not None,
        )
        encoder.embeddings.patch_embeddings.projection = new_proj
        encoder.embeddings.patch_embeddings.num_channels = in_chans
        encoder.config.num_channels = in_chans
    return encoder


def encoder_hidden_size(encoder) -> int:
    return int(encoder.config.hidden_size)
