# -*- coding: utf-8 -*-
import os
import yaml
import random
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from train.tools.data.table.TableCellDataGenerator import TableCellDataGenerator, TableData
from train.tools.data.table.PubTabNetDataset import PubTabNetLMDBDataset

from train.tools.schedulers.CustomCyclicLR import CustomCyclicLR

from train.core import TableStructModel, bbox_to_features

from train.core.common.util import print_network
from train.core import save_model_and_optimizer, load_model_and_optimizer


# =========================
# Dataset
# =========================

class TableStructDataset(Dataset):
    """
    On-the-fly dataset that uses a generator to synthesize tables.
    Each item corresponds to one table; tokens are merged cells.
    """

    def __init__(
            self,
            generator,  # instance of TableCellDataGenerator
            length: int = 10000,  # virtual length (number of tables)
            R_cap: int = 16,  # max classes for rows
            C_cap: int = 32,  # max classes for cols
            use_span_bucket: bool = False,  # if True, bucketize spans (1,2,3,>=4)
            seed: Optional[int] = None,
    ):
        self.gen = generator
        self.length = length
        self.R_cap = R_cap
        self.C_cap = C_cap
        self.use_span_bucket = use_span_bucket
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def __len__(self):
        return self.length

    def _bucketize(self, v: int) -> int:
        # Map span to classes: 1->0, 2->1, 3->2, >=4->3
        if v <= 1:
            return 0
        elif v == 2:
            return 1
        elif v == 3:
            return 2
        else:
            return 3

    def __getitem__(self, idx: int) -> TableData:

        while True:
            data = self.gen.generate(idx)
            # Check the number of rows and cols are in valid range
            if data['meta']['rows'] > self.R_cap or data['meta']['rows'] > self.C_cap:
                continue
            else:
                break

        cells = data["cells"]
        meta = data["meta"]

        # number of row groups is len(meta["row_groups"])
        valid_rows = len(meta.get("row_groups", []))
        valid_cols = meta["cols"]  # number of atom columns (cap for col_id classes)

        feats = []
        row_id = []
        col_id = []
        row_span = []
        col_span = []

        for cell in cells:
            feats.append(bbox_to_features(cell))
            rid = int(cell["row_id"])
            cid = int(cell["col_id"])
            rsp = int(cell["row_span"])
            csp = int(cell["col_span"])

            row_id.append(rid)
            col_id.append(cid)
            if self.use_span_bucket:
                row_span.append(self._bucketize(rsp))
                col_span.append(self._bucketize(csp))
            else:
                # direct span class starting from 1 -> shift to start from 0
                row_span.append(max(1, rsp) - 1)
                col_span.append(max(1, csp) - 1)

        feats = torch.tensor(feats, dtype=torch.float32)
        row_id = torch.tensor(row_id, dtype=torch.long)
        col_id = torch.tensor(col_id, dtype=torch.long)
        row_span = torch.tensor(row_span, dtype=torch.long)
        col_span = torch.tensor(col_span, dtype=torch.long)

        return TableData(
            feats=feats,
            row_id=row_id,
            col_id=col_id,
            row_span=row_span,
            col_span=col_span,
            valid_rows=valid_rows,
            valid_cols=valid_cols
        )


# =========================
# Collate Function
# =========================

def collate_fn(samples: List[TableData], R_cap: int, C_cap: int,
               max_row_span_cls: int, max_col_span_cls: int) -> Dict[str, torch.Tensor]:
    """
    Pads variable-length token sequences and builds masks for multi-class spans.
    """
    B = len(samples)
    N_max = max(s.feats.shape[0] for s in samples)
    D = samples[0].feats.shape[1]

    feats = torch.zeros(B, N_max, D, dtype=torch.float32)
    attn_mask = torch.zeros(B, N_max, dtype=torch.bool)
    row_id = torch.full((B, N_max), -100, dtype=torch.long)
    col_id = torch.full((B, N_max), -100, dtype=torch.long)

    # Revert to multi-class span tensors
    row_span = torch.full((B, N_max), -100, dtype=torch.long)
    col_span = torch.full((B, N_max), -100, dtype=torch.long)

    # Class masks for logits
    row_class_mask = torch.full((B, R_cap), False, dtype=torch.bool)
    col_class_mask = torch.full((B, C_cap), False, dtype=torch.bool)
    row_span_mask = torch.full((B, max_row_span_cls), False, dtype=torch.bool)
    col_span_mask = torch.full((B, max_col_span_cls), False, dtype=torch.bool)

    for i, s in enumerate(samples):
        n = s.feats.shape[0]
        feats[i, :n] = s.feats
        attn_mask[i, :n] = True
        row_id[i, :n] = s.row_id
        col_id[i, :n] = s.col_id
        row_span[i, :n] = s.row_span
        col_span[i, :n] = s.col_span

        row_class_mask[i, :min(s.valid_rows, R_cap)] = True
        col_class_mask[i, :min(s.valid_cols, C_cap)] = True

        # Span masks are based on max span classes, which are constant for the batch
        row_span_mask[i, :max_row_span_cls] = True
        col_span_mask[i, :max_col_span_cls] = True

    batch = {
        "feats": feats,
        "attn_mask": attn_mask,
        "row_id": row_id,
        "col_id": col_id,
        "row_span": row_span,
        "col_span": col_span,
        "row_class_mask": row_class_mask,
        "col_class_mask": col_class_mask,
        "row_span_mask": row_span_mask,
        "col_span_mask": col_span_mask,
    }
    return batch


# =========================
# Loss & Training Utils
# =========================

def masked_ce_loss(logits: torch.Tensor, targets: torch.Tensor,
                   valid_mask: torch.Tensor) -> torch.Tensor:
    """
    Cross-Entropy with token valid mask. targets: [-100] ignored where invalid.
    """
    B, N, C = logits.shape
    logits = logits.reshape(B * N, C)
    targets = targets.reshape(B * N)
    valid = targets != -100
    if valid_mask is not None:
        valid = valid & valid_mask.reshape(B * N)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return F.cross_entropy(logits[valid], targets[valid])


def apply_class_mask(logits: torch.Tensor, class_mask: torch.Tensor) -> torch.Tensor:
    """
    Mask logits so that invalid classes get a very negative score.
      logits: [B, N, C]
      class_mask: [B, C] (True for valid classes)
    """
    B, N, C = logits.shape
    mask = class_mask.unsqueeze(1).expand(B, N, C)
    logits = logits.masked_fill(~mask, -1e9)
    return logits


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    """
    모델의 정확도를 평가합니다.
    """
    model.eval()
    total = {"row": 0.0, "col": 0.0, "rsp": 0.0, "csp": 0.0}
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            feats = batch["feats"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            row_id = batch["row_id"].to(device)
            col_id = batch["col_id"].to(device)
            row_span = batch["row_span"].to(device)
            col_span = batch["col_span"].to(device)

            row_class_mask = batch["row_class_mask"].to(device)
            col_class_mask = batch["col_class_mask"].to(device)
            row_span_mask = batch["row_span_mask"].to(device)
            col_span_mask = batch["col_span_mask"].to(device)

            out = model(feats, attn_mask)
            row_logits = apply_class_mask(out["row_logits"], row_class_mask)
            col_logits = apply_class_mask(out["col_logits"], col_class_mask)
            rsp_logits = apply_class_mask(out["rsp_logits"], row_span_mask)
            csp_logits = apply_class_mask(out["csp_logits"], col_span_mask)

            # 예측된 클래스 ID를 가져옵니다.
            # `argmax(dim=-1)`를 사용하여 가장 높은 확률을 가진 클래스를 선택합니다.
            pred_row_id = torch.argmax(row_logits, dim=-1)
            pred_col_id = torch.argmax(col_logits, dim=-1)
            pred_rsp = torch.argmax(rsp_logits, dim=-1)
            pred_csp = torch.argmax(csp_logits, dim=-1)

            # 정확도를 계산합니다.
            # 예측값과 실제값이 일치하는지 확인하고, 이를 마스크(attn_mask)를 적용해 유효한 토큰에 대해서만 합산합니다.
            row_correct = (pred_row_id == row_id).float()
            col_correct = (pred_col_id == col_id).float()
            rsp_correct = (pred_rsp == row_span).float()
            csp_correct = (pred_csp == col_span).float()

            token_valid = attn_mask.float()
            num_valid_tokens = token_valid.sum()

            if num_valid_tokens > 0:
                total["row"] += (row_correct * token_valid).sum().item()
                total["col"] += (col_correct * token_valid).sum().item()
                total["rsp"] += (rsp_correct * token_valid).sum().item()
                total["csp"] += (csp_correct * token_valid).sum().item()
                total_samples += num_valid_tokens.item()

    if total_samples == 0:
        return {"row": 0.0, "col": 0.0, "rsp": 0.0, "csp": 0.0}

    for k in total:
        total[k] /= total_samples

    return total


# =========================
# Decode (Inference)
# =========================

@torch.no_grad()
def decode_indices(model: nn.Module, feats: torch.Tensor, attn_mask: torch.Tensor,
                   row_class_mask: torch.Tensor, col_class_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Predict row/col indices and spans for a single batch.
      feats: [B, N, D], attn_mask: [B, N]
      row_class_mask: [B, R_cap], col_class_mask: [B, C_cap]
    Returns argmax predictions.
    """
    out = model(feats, attn_mask)
    row_logits = apply_class_mask(out["row_logits"], row_class_mask)
    col_logits = apply_class_mask(out["col_logits"], col_class_mask)

    row_pred = row_logits.argmax(dim=-1)
    col_pred = col_logits.argmax(dim=-1)
    rsp_pred = out["rsp_logits"].argmax(dim=-1)
    csp_pred = out["csp_logits"].argmax(dim=-1)
    return {
        "row_id": row_pred,
        "col_id": col_pred,
        "row_span": rsp_pred,
        "col_span": csp_pred
    }


def main_train(cfg):
    """
    End-to-end training using the given generator, with parameters from a config.
    """

    # Load parameters from config
    train_cfg = cfg['train_table_structure']
    use_span_bucket = False

    epochs = train_cfg['epochs']
    log_step = train_cfg['log_step']
    save_step = train_cfg['save_step']
    eval_step = train_cfg['eval_step']

    num_workers = train_cfg['num_workers']
    save_dir = train_cfg['save_dir']
    load_from = train_cfg['load_from']

    batch_size = train_cfg['batch_size']

    d_model = train_cfg['model']['d_model']
    n_heads = train_cfg['model']['n_heads']
    n_layers = train_cfg['model']['n_layers']
    dropout = train_cfg['model']['dropout']

    device = train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pubtabnet_train_lmdb = train_cfg['pubtabnet']['train_lmdb_path']
    pubtabnet_test_lmdb = train_cfg['pubtabnet']['test_lmdb_path']

    max_row = train_cfg['data_gen']['max_row']
    max_col = train_cfg['data_gen']['max_col']
    span_prob = train_cfg['data_gen']['span_prob']
    max_row_span = train_cfg['data_gen']['max_row_span']
    max_col_span = train_cfg['data_gen']['max_col_span']

    line_count = train_cfg['data_gen']['line_count']
    line_height = train_cfg['data_gen']['line_height']
    line_space = train_cfg['data_gen']['line_space']

    # Virtual dataset size
    train_len = 10000
    val_len = 100

    generator = TableCellDataGenerator(
        row_num=(2, max_row),
        col_num=(2, max_col),
        span_prob=span_prob,
        max_row_span=max_row_span,  # Enabled row merges
        max_col_span=max_col_span,  # Enabled col merges
        line_count=line_count,  # Multi-line text
        line_height=line_height,  # Relative to cell height
        line_spacing=line_space,
        line_width=(0.55, 0.95),
        h_align_mode="fix",
        h_align_probs=(0.6, 0.4),
        v_align_mode="fix",
        v_align_probs=(0.6, 0.4),
        v_align_for_h_center="top",
        empty_cell_prob=0.15,
        padding=(0.03, 0.04),
        jitter=0.005,
    )

    # Span class sizes
    if use_span_bucket:
        S_row = 4
        S_col = 4
    else:
        # Direct classes: 1..max_row_span/max_col_span -> shift to 0..(max-1)
        S_row = generator.max_row_span
        S_col = generator.max_col_span

    # Build datasets/loaders
    gen_train = TableStructDataset(generator, length=train_len, R_cap=max_row, C_cap=max_col,
                                   use_span_bucket=use_span_bucket)
    gen_val = TableStructDataset(generator, length=val_len, R_cap=max_row, C_cap=max_col,
                                 use_span_bucket=use_span_bucket)
    pubtabnet_train = PubTabNetLMDBDataset(lmdb_path=pubtabnet_train_lmdb)
    pt_train = TableStructDataset(pubtabnet_train, length=len(pubtabnet_train), R_cap=max_row, C_cap=max_col,
                                  use_span_bucket=use_span_bucket)
    pubtabnet_val = PubTabNetLMDBDataset(lmdb_path=pubtabnet_test_lmdb)
    pt_val = TableStructDataset(pubtabnet_val, length=len(pubtabnet_val), R_cap=max_row, C_cap=max_col,
                                use_span_bucket=use_span_bucket)

    # train_ds = CombinedDataset(datasets=[pt_train, gen_train], weights=[0.5, 0.5])
    train_ds = gen_train
    # val_ds = pubtab_test
    val_ds = gen_val

    D_in = 14  # feature dimension crafted in bbox_to_features

    def _collate(batch):
        return collate_fn(batch, max_row, max_col, S_row, S_col)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=_collate, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=_collate, pin_memory=True)

    # Build model
    model = TableStructModel(d_in=D_in, R_cap=max_row, C_cap=max_col, S_row=S_row, S_col=S_col,
                             d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                             dropout=dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)

    base_lr = float(cfg['train_table_structure']['scheduler']['base_lr'])
    max_lr = float(cfg['train_table_structure']['scheduler']['max_lr'])
    step_size = int(cfg['train_table_structure']['scheduler']['step_size'])
    gamma = 1.0
    scheduler = CustomCyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size, decay_factor=gamma)

    print_network(model, verbose=True)
    if load_from is not None and os.path.exists(load_from):
        print(f"Loading model from {load_from}...")
        load_model_and_optimizer(model, load_from, optimizer=optimizer)
        print("Model loaded successfully!")

    global_step = 0
    os.makedirs(save_dir, exist_ok=True)

    for ep in range(1, epochs + 1):
        # ---- Train (inline with log_step prints) ----
        model.train()
        total = {"loss": 0.0, "row": 0.0, "col": 0.0, "rsp": 0.0, "csp": 0.0}
        count = 0

        for bidx, batch in enumerate(train_loader, start=1):
            feats = batch["feats"].to(device)
            attn_mask = batch["attn_mask"].to(device)

            row_id = batch["row_id"].to(device)
            col_id = batch["col_id"].to(device)
            row_span = batch["row_span"].to(device)
            col_span = batch["col_span"].to(device)

            row_class_mask = batch["row_class_mask"].to(device)
            col_class_mask = batch["col_class_mask"].to(device)
            row_span_mask = batch["row_span_mask"].to(device)
            col_span_mask = batch["col_span_mask"].to(device)

            out = model(feats, attn_mask)
            row_logits = apply_class_mask(out["row_logits"], row_class_mask)
            col_logits = apply_class_mask(out["col_logits"], col_class_mask)
            rsp_logits = apply_class_mask(out["rsp_logits"], row_span_mask)
            csp_logits = apply_class_mask(out["csp_logits"], col_span_mask)

            token_valid = attn_mask

            loss_row = masked_ce_loss(row_logits, row_id, token_valid)
            loss_col = masked_ce_loss(col_logits, col_id, token_valid)
            loss_rsp = masked_ce_loss(rsp_logits, row_span, token_valid)
            loss_csp = masked_ce_loss(csp_logits, col_span, token_valid)

            loss = loss_row + loss_col + (0.5 * (loss_rsp + loss_csp))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            # running stats
            total["loss"] += float(loss.item())
            total["row"] += float(loss_row.item())
            total["col"] += float(loss_col.item())
            total["rsp"] += float(loss_rsp.item())
            total["csp"] += float(loss_csp.item())
            count += 1
            global_step += 1

            # print per log_step
            if (global_step % log_step) == 0:
                pred_row_id = torch.argmax(row_logits, dim=-1)
                pred_col_id = torch.argmax(col_logits, dim=-1)
                pred_rsp = torch.argmax(rsp_logits, dim=-1)
                pred_csp = torch.argmax(csp_logits, dim=-1)

                row_correct = (pred_row_id == row_id).float()
                col_correct = (pred_col_id == col_id).float()
                rsp_correct = (pred_rsp == row_span).float()
                csp_correct = (pred_csp == col_span).float()
                num_valid_tokens = token_valid.sum()

                row_acc = (row_correct * token_valid).sum().item() / num_valid_tokens.item()
                col_acc = (col_correct * token_valid).sum().item() / num_valid_tokens.item()
                rsp_acc = (rsp_correct * token_valid).sum().item() / num_valid_tokens.item()
                csp_acc = (csp_correct * token_valid).sum().item() / num_valid_tokens.item()

                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {global_step}] lr={current_lr:.2e} | "
                    f"loss={loss.item():.4e} | "
                    f"(row={row_acc:.4f}, col={col_acc:.4f}, rsp={rsp_acc:.4f}, csp={csp_acc:.4f})"
                )

            # save checkpoint every save_step
            if (global_step % save_step) == 0:
                step_path = os.path.join(save_dir, "latest_save_point.pt")
                save_model_and_optimizer(model, optimizer, step_path)
                print(f"[Ep {ep:02d} | step {global_step}] Saved step checkpoint: {os.path.basename(step_path)}")

            # evaluate and save checkpoint every eval_step
            if (global_step % eval_step) == 0:
                print(f"--- [Ep {ep:02d} | step {global_step}] Running evaluation... ---")
                test_result = evaluate(model, val_loader, device)
                print(f"[Ep {ep:02d} | step {global_step}] row_acc: {test_result['row']:.2f} col_acc: {test_result['col']:.2f}"
                      f" rsp_acc: {test_result['rsp']:.2f} csp_acc: {test_result['csp']:.2f}")

                avg_acc = (test_result['row'] + test_result['col']) / 2

                # save checkpoint after evaluation
                step_path = os.path.join(save_dir, f"{avg_acc:.4f}_step_{global_step}.pt")
                save_model_and_optimizer(model, optimizer, step_path)
                print(f"[Ep {ep:02d} | step {global_step}] Saved evaluation checkpoint: {os.path.basename(step_path)}")
                model.train()

    return model


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    with open('./tools/config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    model = main_train(cfg)