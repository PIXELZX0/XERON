#!/usr/bin/env python3
"""XERON decision model with an independently-sized head (wider than the encoder).

Why this exists
---------------
laya's DecisionModel hard-codes the head width to the encoder hidden size:

    d = encoder.config.hidden_size          # 768 for mmBERT-base
    layer = nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout, ...)
    self.head = nn.TransformerEncoder(layer, head_layers)

so "make the head bigger" is impossible without either a new encoder (mmBERT has no
`large`) or a projection in front of the head. This module adds that projection and
rebuilds head / type_emb / scorer / act_head at `head_size`.

When head_size is None (or equals the encoder width) the class behaves exactly like
laya's DecisionModel, so existing checkpoints still load.

Extra capacity is tiny relative to the encoder (per head layer: 4*d^2 attn + 8*d^2 FFN
tokens), e.g. head_size 1024 x 4 layers ~ 50M on top of a 307M encoder.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from laya.common import DecisionModel  # noqa: E402


class WideHeadDecisionModel(DecisionModel):
    """laya DecisionModel + optional head widening via a linear projection.

    Extra kwargs:
        head_size: width of the decision head (default: encoder hidden size)
    """

    def __init__(self, encoder, head_layers=2, head_size=None, n_act=2, dropout=0.1):
        d_enc = encoder.config.hidden_size
        head_size = int(head_size or d_enc)

        # Build laya's model first so all non-head weights (encoder, buffers) are identical.
        super().__init__(encoder, head_layers, n_act, dropout)

        self.head_width = head_size
        self.head_proj = None
        if head_size != d_enc:
            d = head_size
            nhead = max(1, d // 64)
            self.head_proj = nn.Linear(d_enc, d)
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout,
                                               batch_first=True, norm_first=True)
            self.head = (nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
                         if head_layers > 0 else None)
            self.type_emb = nn.Embedding(3, d)
            self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
            self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(), nn.Linear(256, n_act))
            self.temperature = nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                detach_encoder: bool = False):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()
        if self.head_proj is not None:
            h = self.head_proj(h)
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)

        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))
        return logits, act_logits


def build_wide_model(cfg: dict, encoder_dir: str, head_layers: int = 2,
                     head_size: int = None, dropout: float = 0.1):
    """Build WideHeadDecisionModel from a saved encoder directory.

    Mirrors laya.common.build_model but keeps the encoder config intact (so
    ensure_long_context's edits still apply) and honours head_size.
    """
    from transformers import AutoConfig, AutoModel

    ecfg = AutoConfig.from_pretrained(encoder_dir)
    enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    return WideHeadDecisionModel(enc, head_layers=head_layers, head_size=head_size,
                                 n_act=len(cfg.get("act_costs", {})) + 1, dropout=dropout)
