from __future__ import annotations

import torch
import torch.nn.functional as F


def pixel_shuffle_3d(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    if upscale_factor == 1:
        return x
    b, c, d, h, w = x.shape
    r = upscale_factor
    r3 = r ** 3
    if c % r3 != 0:
        raise ValueError(
            f"Channel dimension {c} is not divisible by upscale_factor ** 3 ({r3})."
        )
    x = x.view(b, c // r3, r, r, r, d, h, w)
    x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return x.view(b, c // r3, d * r, h * r, w * r)


def pixel_unshuffle_3d(x: torch.Tensor, downscale_factor: int) -> torch.Tensor:
    if downscale_factor == 1:
        return x
    b, c, d, h, w = x.shape
    r = downscale_factor
    if d % r != 0 or h % r != 0 or w % r != 0:
        raise ValueError(
            f"Spatial dims {(d, h, w)} are not divisible by downscale_factor {r}."
        )
    x = x.view(b, c, d // r, r, h // r, r, w // r, r)
    x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    r3 = r ** 3
    return x.view(b, c * r3, d // r, h // r, w // r)


def upsample_to_reference(
    features: torch.Tensor,
    reference: torch.Tensor,
    mode: str = "trilinear",
) -> torch.Tensor:
    target_size = reference.shape[-3:]
    return F.interpolate(features, size=target_size, mode=mode, align_corners=False)


