import torch
import random
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.patches import Rectangle

@dataclass
class TableData:
    feats: torch.Tensor  # [N, D]
    row_id: torch.Tensor  # [N] (int64)
    col_id: torch.Tensor  # [N] (int64)
    row_span: torch.Tensor  # [N] (int64)
    col_span: torch.Tensor  # [N] (int64)
    valid_rows: int  # actual number of row groups in this table
    valid_cols: int  # actual number of columns in this table


@dataclass
class CellItem:
    # Labels and geometry for each merged cell
    row_id: int
    col_id: int
    row_span: int
    col_span: int
    cell_bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in [0,1]
    has_text: bool
    text_bboxes: Optional[List[Tuple[float, float, float, float]]]  # multi-line text boxes (top->bottom)
    h_align_used: Optional[str] = None  # "left" or "center"
    v_align_used: Optional[str] = None  # "top" or "middle"


class TableCellDataGenerator:
    def __init__(
            self,
            row_num: Tuple[int, int],
            col_num: Tuple[int, int],
            max_row_span: int,
            max_col_span: int,
            span_prob: float = 0.35,
            line_count: Tuple[int, int] = (1, 5),
            line_height: Tuple[float, float] = (0.015, 0.04),  # (min, max) height per line in [0,1]
            line_spacing: float = 0.005,  # vertical gap between lines in [0,1]
            line_width: Tuple[float, float] = (0.05, 0.15),
            # Alignment control
            h_align_mode: str = "mix",  # "mix" | "fix"
            h_align_probs: Tuple[float, float] = (0.6, 0.4),  # (p_left, p_center)
            v_align_mode: str = "mix",  # "mix" | "fix"
            v_align_probs: Tuple[float, float] = (0.6, 0.4),  # (p_top, p_middle)
            v_align_for_h_center: str = "top",  # vertical alignment when horizontal alignment is center
            empty_cell_prob: float = 0.2,
            seed: Optional[int] = None,
            padding: Tuple[float, float] = (0.02, 0.02),
            jitter: float = 0.005,  # jitter is now an absolute value
    ):
        """
        Args:
            row_num: (min_rows, max_rows) inclusive range.
            col_num: (min_cols, max_cols) inclusive range.
            max_row_span: maximum vertical span for merged cells (applies to row groups).
            max_col_span: maximum horizontal span for merged cells.
            span_prob: Initial probability for a cell to have a span greater than 1.
            line_count: (min, max) number of text lines in non-empty cells.
            line_height: (min, max) absolute height per line in [0,1].
            line_spacing: vertical gap between lines in [0,1].
            line_width: (min, max) absolute width per line in [0,1].
            h_align_mode: "mix" for per-column alignment or "fix" for global alignment.
            h_align_probs: when h_align_mode="fix", probabilities for ("left", "center").
            v_align_mode: "mix" for per-row alignment or "fix" for global alignment.
            v_align_probs: when v_align_mode="fix", probabilities for ("top", "middle").
            v_align_for_h_center: The fixed vertical alignment to use when horizontal alignment is 'center'.
            empty_cell_prob: probability that a merged cell has no text.
            seed: RNG seed.
            padding: inner padding ratio per cell (x, y).
            jitter: small random offsets as an absolute value.
        """
        assert isinstance(row_num, tuple) and len(row_num) == 2 and 1 <= row_num[0] <= row_num[1]
        assert isinstance(col_num, tuple) and len(col_num) == 2 and 1 <= col_num[0] <= col_num[1]
        assert max_row_span >= 1 and max_col_span >= 1
        assert 0.0 <= span_prob <= 1.0
        assert 0.0 <= empty_cell_prob <= 1.0
        assert 0.0 <= line_width[0] <= line_width[1] <= 1.0
        assert 0.0 <= line_height[0] <= line_height[1] <= 1.0
        assert 1 <= line_count[0] <= line_count[1]
        assert h_align_mode in ("fix", "mix")
        if h_align_mode == "fix":
            pl, pc = h_align_probs
            assert 0.0 <= pl <= 1.0 and 0.0 <= pc <= 1.0 and abs((pl + pc) - 1.0) < 1e-6
        assert v_align_mode in ("fix", "mix")
        if v_align_mode == "fix":
            pt, pm = v_align_probs
            assert 0.0 <= pt <= 1.0 and 0.0 <= pm <= 1.0 and abs((pt + pm) - 1.0) < 1e-6
        assert v_align_for_h_center in ("top", "middle")

        self.row_range = row_num
        self.col_range = col_num
        self.max_row_span = max_row_span
        self.max_col_span = max_col_span
        self.span_prob = span_prob
        self.line_count = line_count
        self.line_height = line_height
        self.line_spacing = line_spacing
        self.h_align_mode = h_align_mode
        self.h_align_probs = h_align_probs
        self.v_align_mode = v_align_mode
        self.v_align_probs = v_align_probs
        self.v_align_for_h_center = v_align_for_h_center
        self.line_width = line_width
        self.empty_cell_prob = empty_cell_prob
        self.padding = padding
        self.jitter = jitter
        self.rng = random.Random(seed) if seed is not None else random.Random()

    def generate(self, idx=-1) -> Dict[str, Any]:
        # 1) Sample row/col counts
        row_num = self.rng.randint(self.row_range[0], self.row_range[1])
        col_num = self.rng.randint(self.col_range[0], self.col_range[1])

        # 2) Build uniform grid in [0,1]
        col_w = 1.0 / col_num
        row_h = 1.0 / row_num

        def cell_rect(r: int, c: int) -> Tuple[float, float, float, float]:
            # Return atom cell bbox (x1, y1, x2, y2)
            x1 = c * col_w
            y1 = r * row_h
            x2 = (c + 1) * col_w
            y2 = (r + 1) * row_h
            return (x1, y1, x2, y2)

        # Sample alignment types for the entire table based on mode
        if self.h_align_mode == "fix":
            global_h_align = self._choose_h_align()
            col_h_alignments = [global_h_align] * col_num
        else:  # mix
            col_h_alignments = [self._choose_h_align() for _ in range(col_num)]

        if self.v_align_mode == "fix":
            # Apply h_align constraint to global v_align choice
            global_v_align = self._choose_v_align(h_align=col_h_alignments[0])
            row_v_alignments = [global_v_align] * row_num
        else:  # mix
            row_v_alignments = [self._choose_v_align(h_align=col_h_alignments[0]) for _ in range(row_num)]

        # 3) Sample global row groups (vertical bands) to enforce consistent vertical merges
        row_groups = self._sample_row_groups(row_num, self.max_row_span)

        # 4) Tile each row group horizontally with col-span
        items: List[CellItem] = []
        atom_to_merged = [[None] * col_num for _ in range(row_num)]

        for g_start, g_span in row_groups:
            # Use vertical alignment of the first row in the merged cell
            v_align_for_cell = row_v_alignments[g_start]

            c = 0
            while c < col_num:
                cspan = self._sample_span(1, min(self.max_col_span, col_num - c))
                # Use horizontal alignment of the first column in the merged cell
                h_align_for_cell = col_h_alignments[c]

                # Merged cell bbox
                x1, y1, _x2, _y2 = cell_rect(g_start, c)
                x2 = (c + cspan) * col_w
                y2 = (g_start + g_span) * row_h
                cell_bbox = (x1, y1, x2, y2)

                has_text = self.rng.random() >= self.empty_cell_prob
                if has_text:
                    n_lines = self.rng.randint(self.line_count[0], self.line_count[1])
                    text_bboxes = self._sample_multiline_bboxes_inside(
                        cell_bbox, n_lines, h_align_for_cell, v_align_for_cell
                    )
                else:
                    h_align_for_cell = None
                    v_align_for_cell = None
                    text_bboxes = None

                idx = len(items)
                items.append(CellItem(
                    row_id=g_start,
                    col_id=c,
                    row_span=g_span,
                    col_span=cspan,
                    cell_bbox=cell_bbox,
                    has_text=has_text,
                    text_bboxes=text_bboxes,
                    h_align_used=h_align_for_cell,
                    v_align_used=v_align_for_cell
                ))

                for rr in range(g_start, g_start + g_span):
                    for cc in range(c, c + cspan):
                        atom_to_merged[rr][cc] = idx

                c += cspan

        return {
            "meta": {
                "rows": row_num,
                "cols": col_num,
                "col_width": col_w,
                "row_height": row_h,
                "row_groups": [{"start": s, "span": sp} for (s, sp) in row_groups],
                "row_v_alignments": row_v_alignments,
                "col_h_alignments": col_h_alignments,
                "params": {
                    "row_range": self.row_range,
                    "col_range": self.col_range,
                    "max_row_span": self.max_row_span,
                    "max_col_span": self.max_col_span,
                    "span_prob": self.span_prob,
                    "line_count": self.line_count,
                    "line_height": self.line_height,
                    "line_spacing": self.line_spacing,
                    "h_align_mode": self.h_align_mode,
                    "h_align_probs": self.h_align_probs,
                    "v_align_mode": self.v_align_mode,
                    "v_align_probs": self.v_align_probs,
                    "v_align_for_h_center": self.v_align_for_h_center,
                    "line_width": self.line_width,
                    "empty_cell_prob": self.empty_cell_prob,
                    "padding": self.padding,
                    "jitter": self.jitter,
                },
            },
            "cells": [asdict(it) for it in items],
            "atom_to_merged": atom_to_merged,
        }

    # ---------- Internal utilities ----------
    def _choose_h_align(self) -> str:
        pl, pc = self.h_align_probs
        return "left" if self.rng.random() < pl else "center"

    def _choose_v_align(self, h_align: str) -> str:
        if h_align == "center":
            return self.v_align_for_h_center
        pt, pm = self.v_align_probs
        return "top" if self.rng.random() < pt else "middle"

    def _sample_row_groups(self, row_num: int, max_row_span: int) -> List[Tuple[int, int]]:
        groups: List[Tuple[int, int]] = []
        r = 0
        while r < row_num:
            rspan = self._sample_span(1, min(max_row_span, row_num - r))
            groups.append((r, rspan))
            r += rspan
        return groups

    def _sample_span(self, min_v: int, max_v: int) -> int:
        if max_v <= min_v:
            return min_v

        if self.rng.random() > self.span_prob:
            return 1

        span = 2
        p_grow = 0.6
        while span < max_v and self.rng.random() < p_grow:
            span += 1
            p_grow *= 0.6
        return span

    def _inner_region(self, cell_bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = cell_bbox
        cw = max(1e-6, x2 - x1)
        ch = max(1e-6, y2 - y1)
        pad_x = min(self.padding[0], 0.49) * cw
        pad_y = min(self.padding[1], 0.49) * ch
        gx1, gy1 = x1 + pad_x, y1 + pad_y
        gx2, gy2 = x2 - pad_x, y2 - pad_y
        midx, midy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        gx1, gy1 = min(gx1, midx), min(gy1, midy)
        gx2, gy2 = max(gx2, midx), max(gy2, midy)
        return gx1, gy1, gx2, gy2

    def _uniform(self, a: float, b: float) -> float:
        if a == b:
            return a
        return a + (b - a) * self.rng.random()

    def _sample_multiline_bboxes_inside(
            self,
            cell_bbox: Tuple[float, float, float, float],
            n_lines: int,
            h_align: str,
            v_align: str,
    ) -> List[Tuple[float, float, float, float]]:
        gx1, gy1, gx2, gy2 = self._inner_region(cell_bbox)
        gw = max(1e-6, gx2 - gx1)
        gh = max(1e-6, gy2 - gy1)

        # Calculate heights for all lines
        hs = [self._uniform(self.line_height[0], self.line_height[1]) for _ in range(n_lines)]
        total_h = sum(hs)
        total_gap = self.line_spacing * (n_lines - 1)
        total_block_h = total_h + total_gap

        # If the total height exceeds the available space, scale down
        if total_block_h > gh:
            scale_factor = gh / total_block_h
            hs = [h * scale_factor for h in hs]
            total_gap *= scale_factor
            total_block_h = gh

        # Determine y_start based on vertical alignment
        if v_align == "top":
            y_start = gy1
        else:  # middle
            y_start = gy1 + (gh - total_block_h) / 2.0

        bboxes: List[Tuple[float, float, float, float]] = []
        cur_y = y_start

        for i, h in enumerate(hs):
            # Sample line width
            w = self._uniform(self.line_width[0], self.line_width[1])
            w = min(gw, w)

            # Determine x_start based on horizontal alignment
            if h_align == "left":
                x1 = gx1
            else:  # center
                x1 = gx1 + (gw - w) / 2.0
            x2 = x1 + w

            y1 = cur_y
            y2 = cur_y + h

            # Apply jitter to all lines EXCEPT the first one
            if i > 0:
                jx = (self.rng.random() * 2 - 1) * self.jitter
                jy = (self.rng.random() * 2 - 1) * self.jitter
                x1 += jx
                x2 += jx
                y1 += jy
                y2 += jy

            # Clamp bboxes to the inner region
            x1 = max(gx1, min(gx2 - 1e-6, x1))
            x2 = min(gx2, max(gx1 + 1e-6, x2))
            y1 = max(gy1, min(gy2 - h, y1))
            y2 = y1 + h

            bboxes.append((x1, y1, x2, y2))
            cur_y = y2 + self.line_spacing

        return bboxes


def visualize_table(data, show_atom_grid=True, show_labels=True, save_path=None, dpi=150):
    """
    Visualize a generated table:
      - Merged cell bbox (colored by has_text)
      - Multiple text bboxes (red rectangles)
      - Optional atom grid (light gray lines)
    """
    meta = data["meta"]
    rows = meta["rows"]
    cols = meta["cols"]
    cells = data["cells"]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Atom grid
    if show_atom_grid:
        for r in range(rows + 1):
            y = r / rows
            ax.plot([0, 1], [y, y], color="#CCCCCC", lw=0.8, alpha=0.6)
        for c in range(cols + 1):
            x = c / cols
            ax.plot([x, x], [0, 1], color="#CCCCCC", lw=0.8, alpha=0.6)

    # Cells
    for cell in cells:
        x1, y1, x2, y2 = cell["cell_bbox"]
        w = max(1e-6, x2 - x1)
        h = max(1e-6, y2 - y1)
        has_text = cell["has_text"]
        face = "#CDEBC8" if has_text else "#EDEDED"
        ax.add_patch(Rectangle((x1, y1), w, h, facecolor=face, edgecolor="#2C2C2C", lw=1.2, alpha=0.9))

        # multiple text boxes
        if has_text and cell["text_bboxes"]:
            for (tx1, ty1, tx2, ty2) in cell["text_bboxes"]:
                tw = max(1e-6, tx2 - tx1)
                th = max(1e-6, ty2 - ty1)
                ax.add_patch(Rectangle((tx1, ty1), tw, th, facecolor="none", edgecolor="#D64545", lw=1.5))

        if show_labels:
            rid = cell["row_id"];
            cid = cell["col_id"]
            rspan = cell["row_span"];
            cspan = cell["col_span"]
            h_align = cell.get("h_align_used", None)
            v_align = cell.get("v_align_used", None)

            h_label = h_align[0].upper() if h_align else "N/A"
            v_label = v_align[0].upper() if v_align else "N/A"

            cx = x1 + w / 2
            cy = y1 + h / 2
            ax.text(cx, cy, f"r{rid},c{cid}\n{rspan}x{cspan}\n{h_label}/{v_label}", ha="center", va="center",
                    fontsize=9, color="#1F2937")

    ax.set_title(f"Table Visualization (rows={rows}, cols={cols})")
    ax.set_xticks([]);
    ax.set_yticks([])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight");
        plt.close(fig)
    else:
        plt.show()


# ---------------- Example usage ----------------
if __name__ == "__main__":
    while True:
        gen_fixed_align = TableCellDataGenerator(
            row_num=(5, 20),
            col_num=(5, 20),
            max_row_span=5,
            max_col_span=5,
            span_prob=0.35,
            line_count=(1, 5),
            line_height=(0.02, 0.04),
            line_spacing=0.0,
            line_width=(0.05, 0.15),
            h_align_mode="fix",
            h_align_probs=(0.6, 0.4),
            v_align_mode="fix",
            v_align_probs=(0.6, 0.4),
            v_align_for_h_center="top",
            empty_cell_prob=0.15,
            padding=(0.03, 0.04),
            jitter=0.005,
        )

        print("Generating table with fixed global alignment (consistent start points)...")
        data_fixed_align = gen_fixed_align.generate()
        print(data_fixed_align)
        visualize_table(data_fixed_align, show_atom_grid=True, show_labels=True)
