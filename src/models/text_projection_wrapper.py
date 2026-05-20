"""Trainable Linear text projection wrapper.

Used by TEXT_ENCODER=sbert_proj / concat_proj experiments. Wraps any backbone
that takes (x, edge_index_dict) and learns an end-to-end nn.Linear(raw_dim ->
proj_dim) projection on a separate raw text view (`text_raw`), then overwrites
the first `proj_dim` columns of `x` with the projection output before calling
the backbone.

Why a wrapper instead of editing each model:
  * Backbones, configs, loss head, CARE filter, build_relations and
    multi-seed launcher all keep their existing signatures.
  * features.npy stays (N, 140) so build_relations[:, :128] still works.
  * The trainable projection is the only new learnable parameter introduced
    on top of the FINAL model — clean ablation against the frozen-SVD variants
    (`sbert` and `concat`).

Leakage-safe: `text_raw` is precomputed from frozen SBERT (and optionally
train-only-fit TF-IDF/SVD) — no label use. The wrapper itself only sees raw
features and learns from the standard fraud loss like any other parameter.
"""
import torch
import torch.nn as nn


class TextProjectionWrapper(nn.Module):
    def __init__(self, backbone, text_raw, raw_dim, proj_dim=128,
                 numeric_start=128, dropout=0.0):
        """
        Args:
            backbone: nn.Module taking (x, edge_index_dict) -> logits or
                      (logits, aux_logits).
            text_raw: torch.FloatTensor of shape (N, raw_dim). Registered as
                      buffer so it moves with .to(device) but is not learnable.
            raw_dim:  text_raw.shape[1]; the Linear projection's input dim.
            proj_dim: output dim of the projection. Must equal the number of
                      text columns in features.npy (default 128 == svd_components).
            numeric_start: the first column of `x` that is NOT text. For our
                      pipeline this is always svd_components (== proj_dim).
            dropout:  optional dropout on the projection output.
        """
        super().__init__()
        assert text_raw.dim() == 2, "text_raw must be (N, raw_dim)"
        assert text_raw.shape[1] == raw_dim, (
            f"text_raw last dim {text_raw.shape[1]} != raw_dim {raw_dim}"
        )
        self.backbone = backbone
        self.proj = nn.Linear(raw_dim, proj_dim)
        self.proj_dim = proj_dim
        self.numeric_start = numeric_start
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # buffer: not a Parameter, but moves with .to(device) and is part of state_dict
        self.register_buffer("text_raw", text_raw)

    def _project_x(self, x):
        text_proj = self.dropout(self.proj(self.text_raw))  # (N, proj_dim)
        if x.shape[1] <= self.numeric_start:
            # No numeric tail; pure-text input. Replace fully.
            return text_proj
        numeric = x[:, self.numeric_start:]                 # (N, num_dim)
        return torch.cat([text_proj, numeric], dim=1)       # (N, proj_dim+num_dim)

    def forward(self, x, edge_index_dict, *args, **kwargs):
        x_new = self._project_x(x)
        return self.backbone(x_new, edge_index_dict, *args, **kwargs)

    # Convenience pass-through for code that introspects the backbone.
    @property
    def wrapped(self):
        return self.backbone
