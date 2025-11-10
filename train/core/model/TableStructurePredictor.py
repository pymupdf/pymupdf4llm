# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any

def bbox_to_features(cell: Dict[str, Any]) -> List[float]:
    """
    Convert a merged cell record into a fixed-size feature vector.
      Geometry (normalized):
        - x1, y1, x2, y2, cx, cy, w, h, aspect
      Text block stats (from multi-line bboxes):
        - line_cnt, mean_line_h, mean_line_w
      Alignment one-hot:
        - align_left, align_center
    """
    x1, y1, x2, y2 = cell["cell_bbox"]
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    aspect = w / h

    if cell["has_text"] and cell["text_bboxes"]:
        lines = cell["text_bboxes"]
        lc = len(lines)
        hs = [max(1e-6, (tb[3]-tb[1])) for tb in lines]
        ws = [max(1e-6, (tb[2]-tb[0])) for tb in lines]
        mean_h = sum(hs) / lc
        mean_w = sum(ws) / lc
    else:
        lc, mean_h, mean_w = 0.0, 0.0, 0.0

    # alignment flags (left/center)
    align_used = cell.get("align_used", None)
    align_left = 1.0 if align_used == "left" else 0.0
    align_center = 1.0 if align_used == "center" else 0.0

    feat = [x1, y1, x2, y2, cx, cy, w, h, aspect,
            float(lc), float(mean_h), float(mean_w),
            align_left, align_center]
    return feat

class TransformerBackbone(nn.Module):
    """
    Simple Transformer encoder for token-wise features.
    """
    def __init__(self, d_in: int, d_model: int = 256, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, D_in]
        attn_mask: [B, N] (True for valid)
        """
        h = self.input_proj(x)  # [B, N, d_model]
        # build key padding mask: True for PAD positions
        key_padding_mask = ~attn_mask  # [B, N]
        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)
        return h  # [B, N, d_model]


class TableHead(nn.Module):
    """
    Multi-head classifier for row_id, col_id, row_span, col_span.
    """
    def __init__(self, d_model: int, R_cap: int, C_cap: int, S_row: int, S_col: int):
        super().__init__()
        hidden = d_model
        self.row_cls = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, R_cap))
        self.col_cls = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, C_cap))
        self.rsp_cls = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, S_row))
        self.csp_cls = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU(), nn.Linear(hidden, S_col))

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "row_logits": self.row_cls(h),  # [B, N, R_cap]
            "col_logits": self.col_cls(h),  # [B, N, C_cap]
            "rsp_logits": self.rsp_cls(h),  # [B, N, S_row]
            "csp_logits": self.csp_cls(h),  # [B, N, S_col]
        }


class TableStructModel(nn.Module):
    def __init__(self, d_in: int, R_cap: int, C_cap: int, S_row: int, S_col: int,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.backbone = TransformerBackbone(d_in, d_model, n_heads, n_layers, dropout)
        self.head = TableHead(d_model, R_cap, C_cap, S_row, S_col)

    def forward(self, feats: torch.Tensor, attn_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(feats, attn_mask)
        out = self.head(h)
        return out

class TableStructurePredictor:
    """
    Load a trained checkpoint and run inference on normalized bbox lists.
    - Expected checkpoint format (as saved in training):
        {
          "model": state_dict,
          "cfg": {"D_in": int, "R_cap": int, "C_cap": int, "S_row": int, "S_col": int}
        }
      If "cfg" is missing, you can override params via __init__ args.
    - Input bboxes must be normalized in [0,1]: (x1, y1, x2, y2)
    """
    # --------- Public API ---------
    def __init__(
        self,
        ckpt_path: str,
        device: Optional[str] = None,
        # Optional overrides if checkpoint lacks "cfg" or differs
        d_in_override: Optional[int] = None,
        R_cap_override: Optional[int] = None,
        C_cap_override: Optional[int] = None,
        S_row_override: Optional[int] = None,
        S_col_override: Optional[int] = None,
        # Span decoding mode: bucketized (1,2,3,>=4) or direct (1..max)
        span_bucket: Optional[bool] = None,  # None -> auto from S_row/S_col (==4 -> bucket)
        # Backbone dims (must match training if you changed them)
        model_dims: Tuple[int, int, int, float] = (256, 8, 4, 0.1),  # (d_model, n_heads, n_layers, dropout)
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj = torch.load(ckpt_path, map_location=self.device)

        # Support both dict with "model" and raw state_dict
        state = obj.get("model", obj)
        cfg = obj.get("cfg", {})

        # Resolve model caps and input dim
        self.D_in = d_in_override if d_in_override is not None else cfg.get("D_in", 14)
        self.R_cap = R_cap_override if R_cap_override is not None else cfg.get("R_cap", 32)
        self.C_cap = C_cap_override if C_cap_override is not None else cfg.get("C_cap", 64)
        self.S_row = S_row_override if S_row_override is not None else cfg.get("S_row", 4)
        self.S_col = S_col_override if S_col_override is not None else cfg.get("S_col", 4)

        # Decide span bucket mode
        if span_bucket is None:
            self.span_bucket = (self.S_row == 4 and self.S_col == 4)
        else:
            self.span_bucket = bool(span_bucket)

        # Build model and load weights
        d_model, n_heads, n_layers, dropout = model_dims
        self.model = self._TableStructModel(
            d_in=self.D_in, R_cap=self.R_cap, C_cap=self.C_cap,
            S_row=self.S_row, S_col=self.S_col,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers, dropout=dropout
        ).to(self.device)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        bboxes: List[Tuple[float, float, float, float]],
        rows_hint: Optional[int] = None,
        cols_hint: Optional[int] = None,
        return_probs: bool = False,
        return_grid: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on a single table's bbox list.
        Args:
            bboxes: list of (x1,y1,x2,y2), normalized to [0,1]
            rows_hint: optional upper bound for row classes to stabilize prediction
            cols_hint: optional upper bound for col classes to stabilize prediction
            return_probs: if True, include softmax probabilities for debugging
            return_grid: if True, return a naive 2D grid filled by predicted spans
        Returns:
            {
              "preds": [
                 { "bbox": (..), "row_id": int, "col_id": int, "row_span": int, "col_span": int,
                   "row_prob": Optional[List[float]], "col_prob": Optional[List[float]],
                   "row_span_prob": Optional[List[float]], "col_span_prob": Optional[List[float]] },
                 ...
              ],
              "grid": Optional[Dict[str, Any]],
              "meta": { "num_tokens": N, "R_cap": ..., "C_cap": ..., "S_row": ..., "S_col": ... }
            }
        """
        # Edge case: empty input
        if not bboxes:
            return {"preds": [], "grid": None, "meta": {"num_tokens": 0, "R_cap": self.R_cap, "C_cap": self.C_cap}}

        # Build features
        feats = self._bboxes_to_features(bboxes).unsqueeze(0).to(self.device)  # [1, N, D]
        N = feats.shape[1]

        # Masks
        masks = self._make_masks(B=1, N=N, rows_hint=rows_hint, cols_hint=cols_hint)
        attn_mask = masks["attn_mask"]
        row_class_mask = masks["row_class_mask"]
        col_class_mask = masks["col_class_mask"]
        row_span_mask = masks["row_span_mask"]
        col_span_mask = masks["col_span_mask"]

        # Forward pass
        out = self.model(feats, attn_mask)
        row_logits = self._apply_class_mask(out["row_logits"], row_class_mask)
        col_logits = self._apply_class_mask(out["col_logits"], col_class_mask)
        rsp_logits = self._apply_class_mask(out["rsp_logits"], row_span_mask)
        csp_logits = self._apply_class_mask(out["csp_logits"], col_span_mask)

        # Argmax predictions
        row_pred = row_logits.argmax(dim=-1)[0].tolist()
        col_pred = col_logits.argmax(dim=-1)[0].tolist()
        rsp_pred = rsp_logits.argmax(dim=-1)[0].tolist()
        csp_pred = csp_logits.argmax(dim=-1)[0].tolist()

        # Optional probabilities
        if return_probs:
            row_prob = torch.softmax(row_logits, dim=-1)[0].cpu().tolist()
            col_prob = torch.softmax(col_logits, dim=-1)[0].cpu().tolist()
            rsp_prob = torch.softmax(rsp_logits, dim=-1)[0].cpu().tolist()
            csp_prob = torch.softmax(csp_logits, dim=-1)[0].cpu().tolist()
        else:
            row_prob = col_prob = rsp_prob = csp_prob = None

        # Pack results
        preds = []
        for i, bb in enumerate(bboxes):
            preds.append({
                "bbox": tuple(map(float, bb)),
                "row_id": int(row_pred[i]),
                "col_id": int(col_pred[i]),
                "row_span": int(self._map_span_pred(rsp_pred[i], is_row=True)),
                "col_span": int(self._map_span_pred(csp_pred[i], is_row=False)),
                "row_prob": row_prob[i] if return_probs else None,
                "col_prob": col_prob[i] if return_probs else None,
                "row_span_prob": rsp_prob[i] if return_probs else None,
                "col_span_prob": csp_prob[i] if return_probs else None,
            })

        grid = self._build_grid(preds) if return_grid else None

        return {
            "preds": preds,
            "grid": grid,
            "meta": {
                "num_tokens": N,
                "R_cap": self.R_cap,
                "C_cap": self.C_cap,
                "S_row": self.S_row,
                "S_col": self.S_col,
                "rows_hint": rows_hint,
                "cols_hint": cols_hint,
            }
        }

    # --------- Internals ---------

    def _bboxes_to_features(self, bboxes: List[Tuple[float, float, float, float]]) -> torch.Tensor:
        """
        Convert raw bboxes to fixed features used at training time.
        Feature layout (D_in must match training):
          [x1,y1,x2,y2,cx,cy,w,h,aspect, line_cnt, mean_line_h, mean_line_w, align_left, align_center]
        For external inference, text-related stats are set to zeros.
        """
        feats = []
        for (x1, y1, x2, y2) in bboxes:
            w = max(1e-6, x2 - x1)
            h = max(1e-6, y2 - y1)
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            aspect = w / h
            feats.append([x1, y1, x2, y2, cx, cy, w, h, aspect,
                          0.0, 0.0, 0.0, 0.0, 0.0])  # zeros for unknown text stats
        x = torch.tensor(feats, dtype=torch.float32) if feats else torch.zeros((0, self.D_in), dtype=torch.float32)
        if x.shape[1] != self.D_in:
            raise ValueError(f"Feature dim mismatch: expected {self.D_in}, got {x.shape[1]}")
        return x

    def _make_masks(self, B: int, N: int, rows_hint: Optional[int], cols_hint: Optional[int]) -> Dict[str, torch.Tensor]:
        """
        Build attention and class masks. Hints cap the valid class ranges for stability.
        """
        attn_mask = torch.zeros(B, N, dtype=torch.bool, device=self.device)
        if N > 0:
            attn_mask[:, :N] = True

        row_class_mask = torch.zeros(B, self.R_cap, dtype=torch.bool, device=self.device)
        col_class_mask = torch.zeros(B, self.C_cap, dtype=torch.bool, device=self.device)
        rmax = min(rows_hint, self.R_cap) if rows_hint is not None else self.R_cap
        cmax = min(cols_hint, self.C_cap) if cols_hint is not None else self.C_cap
        row_class_mask[:, :rmax] = True
        col_class_mask[:, :cmax] = True

        row_span_mask = torch.ones(B, self.S_row, dtype=torch.bool, device=self.device)
        col_span_mask = torch.ones(B, self.S_col, dtype=torch.bool, device=self.device)

        return {
            "attn_mask": attn_mask,
            "row_class_mask": row_class_mask,
            "col_class_mask": col_class_mask,
            "row_span_mask": row_span_mask,
            "col_span_mask": col_span_mask,
        }

    @staticmethod
    def _apply_class_mask(logits: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
        """
        Mask logits so that invalid classes get a very negative score.
        """
        B, N, C = logits.shape
        mask = class_mask.unsqueeze(1).expand(B, N, C)
        return logits.masked_fill(~mask, -1e9)

    def _map_span_pred(self, span_idx: int, is_row: bool) -> int:
        """
        Map predicted span class to span value.
          - Direct: class k -> span = k+1
          - Bucket: 0->1, 1->2, 2->3, 3->4 (interpreted as 4+ in practice)
        """
        if not self.span_bucket:
            return int(span_idx) + 1
        # bucket mode (size=4)
        idx = int(span_idx)
        if idx <= 0:
            return 1
        elif idx == 1:
            return 2
        elif idx == 2:
            return 3
        else:
            return 4  # represents ">=4"

    def _build_grid(self, preds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Construct a naive 2D grid by filling predicted spans.
        This is for inspection only; it does not resolve conflicts.
        """
        if not preds:
            return {"grid": [], "rows": 0, "cols": 0}

        R = max(p["row_id"] + p["row_span"] for p in preds)
        C = max(p["col_id"] + p["col_span"] for p in preds)
        grid = [[None for _ in range(C)] for _ in range(R)]

        for idx, p in enumerate(preds):
            r0, c0 = p["row_id"], p["col_id"]
            rs, cs = p["row_span"], p["col_span"]
            for r in range(r0, min(R, r0 + rs)):
                for c in range(c0, min(C, c0 + cs)):
                    if grid[r][c] is None:
                        grid[r][c] = idx
        return {"grid": grid, "rows": R, "cols": C}
