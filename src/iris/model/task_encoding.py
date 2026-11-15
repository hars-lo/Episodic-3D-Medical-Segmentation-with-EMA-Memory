from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import pixel_shuffle_3d, pixel_unshuffle_3d, upsample_to_reference


class TaskEncodingModule(nn.Module):
    def __init__(
        self,
        *,
        feature_channels: int,
        num_query_tokens: int = 8,
        num_attention_heads: int = 8,
        downsample_ratio: int = 16,
    ) -> None:
        super().__init__()
        if downsample_ratio < 1:
            raise ValueError("downsample_ratio must be >= 1")

        self.downsample_ratio = downsample_ratio
        r_cubed = downsample_ratio ** 3

        if feature_channels % num_attention_heads != 0:
            adjusted_heads = math.gcd(feature_channels, num_attention_heads)
            num_attention_heads = max(1, adjusted_heads)
        self.num_attention_heads = num_attention_heads
        self.num_query_tokens = num_query_tokens

        self.context_channels = max(1, math.ceil(feature_channels / r_cubed))

        self.pre_shuffle = nn.Conv3d(
            feature_channels,
            self.context_channels * r_cubed,
            kernel_size=1,
            bias=False,
        )
        self.context_conv = nn.Conv3d(
            self.context_channels + 1,
            self.context_channels,
            kernel_size=1,
            bias=True,
        )
        self.post_unshuffle = nn.Conv3d(
            self.context_channels * r_cubed,
            feature_channels,
            kernel_size=1,
            bias=False,
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=self.num_attention_heads,
            batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            embed_dim=feature_channels,
            num_heads=self.num_attention_heads,
            batch_first=True,
        )

        self.query_tokens = nn.Parameter(
            torch.randn(1, num_query_tokens, feature_channels)
        )
        nn.init.normal_(self.query_tokens, mean=0.0, std=feature_channels ** -0.5)

    def forward(
        self,
        support_features: torch.Tensor,
        support_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if support_masks.dtype != support_features.dtype:
            support_masks = support_masks.to(support_features.dtype)

        b, c, d, h, w = support_features.shape
        _, k, D, H, W = support_masks.shape

        upsampled = upsample_to_reference(support_features, support_masks)
        upsampled = upsampled.unsqueeze(1).expand(-1, k, -1, -1, -1, -1)
        masks = support_masks.unsqueeze(2)
        masked = upsampled * masks
        mask_sum = masks.sum(dim=(-3, -2, -1)).clamp_min(1e-6)
        pooled = masked.sum(dim=(-3, -2, -1)) / mask_sum
        T_f = pooled.unsqueeze(2) 

        # Contextual encoding pipeline
        expanded = self.pre_shuffle(support_features)
        shuffled = pixel_shuffle_3d(expanded, self.downsample_ratio) 
        shuffled = shuffled.unsqueeze(1).expand(-1, k, -1, -1, -1, -1)
        concat_input = torch.cat([shuffled, masks], dim=2)
        context = concat_input.reshape(b * k, concat_input.shape[2], D, H, W)
        context = self.context_conv(context)
        context = context.view(b * k, self.context_channels, D, H, W)
        context = pixel_unshuffle_3d(context, self.downsample_ratio)
        context = self.post_unshuffle(context)
        context = context.view(b, k, c, d, h, w)

        spatial_tokens = context.reshape(b * k, c, d * h * w).transpose(1, 2)
        query_tokens = self.query_tokens.expand(b * k, -1, -1)

        tokens_after_cross, _ = self.cross_attn(
            query_tokens,
            spatial_tokens,
            spatial_tokens,
        )
        tokens_after_self, _ = self.self_attn(
            tokens_after_cross,
            tokens_after_cross,
            tokens_after_cross,
        )
        T_c = tokens_after_self.view(b, k, -1, c)

        task_embeddings = torch.cat([T_f, T_c], dim=2)
        return {
            "task_embeddings": task_embeddings,
            "foreground_embeddings": T_f,
            "context_tokens": T_c,
        }


