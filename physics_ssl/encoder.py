"""Per-frame 2D ViT encoder for multi-channel physics frames."""

from torch import nn

import stable_pretraining as spt


def build_per_frame_encoder(
    in_chans: int = 11,
    image_size: int = 256,
    patch_size: int = 16,
    scale: str = "tiny",
    pretrained: bool = False,
):
    """Wraps ``spt.backbone.utils.vit_hf`` so it accepts arbitrary input channels.

    HF ``ViTConfig`` supports ``num_channels``; passing it through kwargs makes
    ``ViTPatchEmbeddings`` instantiate ``Conv2d(num_channels, hidden, k, s)``
    directly. We then assert the live ``Conv2d.in_channels`` equals ``in_chans``
    — the config alone is not a reliable invariant because some HF versions
    write the kwarg into config but keep the default 3-channel projection.
    If we detect drift we rewrite the projection in-place, then re-check.

    Returns the bare HF ``ViTModel``. Caller is expected to read
    ``output.last_hidden_state[:, 0]`` for the CLS embedding.
    """
    encoder = spt.backbone.utils.vit_hf(
        scale,
        patch_size=patch_size,
        image_size=image_size,
        pretrained=pretrained,
        use_mask_token=False,
        num_channels=in_chans,
    )
    proj = encoder.embeddings.patch_embeddings.projection
    if getattr(proj, "in_channels", None) != in_chans:
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
        proj = new_proj
    if proj.in_channels != in_chans:
        raise RuntimeError(
            f"Failed to wire ViT for in_chans={in_chans}; patch projection "
            f"still reports in_channels={proj.in_channels}. The HF ViT layout "
            "likely changed; update build_per_frame_encoder to match."
        )
    return encoder


def encoder_hidden_size(encoder) -> int:
    return int(encoder.config.hidden_size)
