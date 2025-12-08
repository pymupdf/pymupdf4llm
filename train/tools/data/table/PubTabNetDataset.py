
import os
import json
import lmdb
import random
import tempfile
import pickle


import matplotlib.pyplot as plt
import matplotlib.patches as patches

from torch.utils.data import Dataset
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from collections import Counter
from dataclasses import asdict

from train.tools.data.table.TableCellDataGenerator import CellItem


class PubTabNetJSONLDataset(Dataset):
    def __init__(self, jsonl_file: str, max_samples=None, line_margin_ratio=(0.0, 0.1), infer_span: bool = True):
        """
        Args:
            jsonl_file (str): Path to PubTabNet jsonl file.
            max_samples (int, optional): Load only first max_samples for testing.
            line_margin_ratio (tuple): Min/max fraction of bbox height to add as margin for multi-line splitting.
            infer_span (bool): If True, infer spans based on empty 'tokens'. If False, use explicit HTML tags.
        """
        self.data = []
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                self.data.append(json.loads(line.strip()))
        self.line_margin_ratio = line_margin_ratio
        self.infer_span = infer_span
        random.seed(42)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.generate(idx)

    def generate(self, idx: int = -1) -> Dict[str, Any]:
        """
        Generates table data for a given index, formatted like TableCellDataGenerator.
        """
        if idx < 0:
            if not self.data:
                return {"meta": {"rows": 0, "cols": 0}, "cells": [], "atom_to_merged": []}
            idx = random.randint(0, len(self.data) - 1)

        entry = self.data[idx]

        if self.infer_span:
            # Use the original, inference-based method
            cells = self._assign_cells_inferred(entry)
        else:
            # Use the revised, explicit HTML parsing method
            cells = self._assign_cells_explicit(entry)

        if not cells:
            return {
                "meta": {"rows": 0, "cols": 0},
                "cells": [],
                "atom_to_merged": []
            }

        x_min = min(c["bbox"][0] for c in cells)
        y_min = min(c["bbox"][1] for c in cells)
        x_max = max(c["bbox"][2] for c in cells)
        y_max = max(c["bbox"][3] for c in cells)

        row_num = max(c["row_idx"] for c in cells) + 1 if cells else 0
        col_num = max(c["col_idx"] for c in cells) + 1 if cells else 0

        items: List[CellItem] = []
        atom_to_merged = [[-1] * col_num for _ in range(row_num)]

        visited_cells = set()

        for cell in cells:
            key = (cell["row_idx"], cell["col_idx"])
            if key in visited_cells:
                continue
            visited_cells.add(key)

            row_span = cell.get("row_span", 1)
            col_span = cell.get("col_span", 1)

            # Normalize bbox to [0,1] based on table boundaries
            nx0 = (cell["bbox"][0] - x_min) / (x_max - x_min + 1e-6)
            ny0 = (cell["bbox"][1] - y_min) / (y_max - y_min + 1e-6)
            nx1 = (cell["bbox"][2] - x_min) / (x_max - x_min + 1e-6)
            ny1 = (cell["bbox"][3] - y_min) / (y_max - y_min + 1e-6)

            normalized_bbox = (nx0, ny0, nx1, ny1)

            has_text = bool(cell.get("tokens"))
            text_bboxes = [normalized_bbox] if has_text else []

            idx = len(items)
            items.append(CellItem(
                row_id=cell["row_idx"],
                col_id=cell["col_idx"],
                row_span=row_span,
                col_span=col_span,
                cell_bbox=normalized_bbox,
                has_text=has_text,
                text_bboxes=text_bboxes,
                h_align_used=None,
                v_align_used=None
            ))

            for r in range(cell["row_idx"], cell["row_idx"] + row_span):
                for c in range(cell["col_idx"], cell["col_idx"] + col_span):
                    if 0 <= r < row_num and 0 <= c < col_num:
                        atom_to_merged[r][c] = idx

        return {
            "meta": {
                "rows": row_num,
                "cols": col_num,
            },
            "cells": [asdict(it) for it in items],
            "atom_to_merged": atom_to_merged,
        }

    # --- Helper methods for explicit HTML parsing (infer_span=False) ---
    def _parse_structure_to_grid_explicit(self, structure_tokens: List[str]) -> Tuple[List[List[Any]], int, int]:
        grid = []
        row_idx = -1
        col_idx = 0

        for token in structure_tokens:
            if token == "<tr>":
                row_idx += 1
                col_idx = 0
                if row_idx >= len(grid):
                    grid.append([])
                continue

            if token.startswith("<td"):
                rowspan = 1
                colspan = 1
                if "rowspan" in token:
                    rowspan = int(token.split('rowspan="')[1].split('"')[0])
                if "colspan" in token:
                    colspan = int(token.split('colspan="')[1].split('"')[0])

                while col_idx < len(grid[row_idx]) and grid[row_idx][col_idx] is not None:
                    col_idx += 1

                for r in range(row_idx, row_idx + rowspan):
                    if r >= len(grid):
                        grid.append([None] * (col_idx))
                    while len(grid[r]) <= col_idx + colspan - 1:
                        grid[r].append(None)

                grid[row_idx][col_idx] = {"colspan": colspan, "rowspan": rowspan}
                for r in range(row_idx, row_idx + rowspan):
                    for c in range(col_idx, col_idx + colspan):
                        if (r, c) != (row_idx, col_idx):
                            grid[r][c] = "placeholder"
                col_idx += colspan

        max_rows = len(grid)
        max_cols = max(len(row) for row in grid) if grid else 0
        return grid, max_rows, max_cols

    def _assign_cells_explicit(self, entry) -> List[Dict[str, Any]]:
        structure_tokens = entry["html"]["structure"]["tokens"]
        raw_cells = entry["html"]["cells"]
        grid_info, rows, cols = self._parse_structure_to_grid_explicit(structure_tokens)

        assigned_cells = []
        cell_idx = 0
        for r in range(rows):
            for c in range(cols):
                if r >= len(grid_info) or c >= len(grid_info[r]) or grid_info[r][c] == "placeholder":
                    continue

                if grid_info[r][c] is None:
                    # An empty cell based on structure, likely a table with missing cells
                    # or an incomplete structure representation.
                    assigned_cells.append({
                        "row_idx": r, "col_idx": c, "row_span": 1, "col_span": 1,
                        "bbox": [0.0, 0.0, 0.0, 0.0], "tokens": [], "has_text": False
                    })
                    continue

                if cell_idx >= len(raw_cells):
                    continue

                cell = raw_cells[cell_idx]
                cell_idx += 1

                span_info = grid_info[r][c]
                cell["row_idx"] = r
                cell["col_idx"] = c
                cell["row_span"] = span_info["rowspan"]
                cell["col_span"] = span_info["colspan"]

                # Check for `tokens` existence before accessing
                cell["has_text"] = bool(cell.get("tokens"))
                if "text_bboxes" not in cell or not cell["text_bboxes"]:
                    if "bbox" in cell and cell["bbox"] is not None:
                        cell["text_bboxes"] = [cell["bbox"]] if cell["has_text"] else []
                    else:
                        cell["bbox"] = [0.0, 0.0, 1.0, 1.0]
                        cell["text_bboxes"] = []

                assigned_cells.append(cell)
        return assigned_cells

    # --- Helper methods for inference-based span (infer_span=True) ---
    def _parse_structure_to_grid_inferred(self, structure_tokens: List[str]) -> List[List[Any]]:
        grid, row = [], []
        for token in structure_tokens:
            if token == "<tr>":
                row = []
            elif token == "</tr>":
                grid.append(row)
            elif token == "<td>":
                row.append(None)
        return grid

    def _assign_cells_inferred(self, entry) -> List[Dict[str, Any]]:
        structure_tokens = entry["html"]["structure"]["tokens"]
        raw_cells = entry["html"]["cells"]
        grid = self._parse_structure_to_grid_inferred(structure_tokens)

        idx = 0
        for r, row in enumerate(grid):
            for c in range(len(row)):
                if idx < len(raw_cells):
                    cell = raw_cells[idx]
                    cell["row_idx"] = r
                    cell["col_idx"] = c
                    cell["row_span"] = 1
                    cell["col_span"] = 1
                    cell["has_text"] = bool(cell.get("tokens"))
                    if "bbox" not in cell or cell["bbox"] is None:
                        cell["bbox"] = [0.0, 0.0, 1.0, 1.0]
                    grid[r][c] = cell
                    idx += 1

        rows = len(grid)
        cols = max(len(row) for row in grid) if grid else 0
        visited = [[False] * cols for _ in range(rows)]

        enriched = []
        for r in range(rows):
            for c in range(len(grid[r])):
                cell = grid[r][c]
                if cell is None or visited[r][c]:
                    continue
                visited[r][c] = True

                if not cell.get("tokens", []):
                    enriched.append(cell)
                    continue

                col_span = 1
                cc = c + 1
                while cc < len(grid[r]) and grid[r][cc] and not grid[r][cc].get("tokens", []):
                    col_span += 1
                    visited[r][cc] = True
                    cc += 1

                row_span = 1
                rr = r + 1
                while rr < rows and c < len(grid[rr]) and grid[rr][c] and not grid[rr][c].get("tokens", []):
                    row_span += 1
                    visited[rr][c] = True
                    rr += 1

                cell["row_span"] = row_span
                cell["col_span"] = col_span
                enriched.append(cell)

        return enriched

    def _find_mode_height(self, heights: List[float]) -> float:
        # (생략: 기존 코드와 동일)
        if not heights:
            return 1.0
        c = Counter([round(h) for h in heights])
        return float(c.most_common(1)[0][0])

    def _split_cell_into_lines(self, bbox, mode_height) -> List[List[float]]:
        # (생략: 기존 코드와 동일)
        x0, y0, x1, y1 = bbox
        h = y1 - y0
        min_margin, max_margin = self.line_margin_ratio
        if h <= 1.2 * mode_height:
            return [(x0, y0, x1, y1)]
        n_lines = max(1, int(round(h / mode_height)))
        lines = []
        for i in range(n_lines):
            line_h = h / n_lines
            margin = random.uniform(min_margin * line_h, max_margin * line_h)
            ly0 = y0 + i * line_h
            ly1 = y0 + (i + 1) * line_h - margin
            if i == n_lines - 1:
                ly1 = y1
            lines.append((x0, ly0, x1, ly1))
        return lines


class PubTabNetLMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        """
        Dataset that loads PubTabNet samples from an LMDB database.

        Args:
            lmdb_path (str): Path to LMDB folder
        """
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=True, meminit=False)
        with self.env.begin() as txn:
            self.length = txn.stat()['entries']

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return self.generate(idx)

    def generate(self, idx):
        with self.env.begin() as txn:
            key = f"{idx:08}".encode('utf-8')
            data_bytes = txn.get(key)
            if data_bytes is None:
                raise IndexError(f"Index {idx} not found in LMDB.")
            data_bytes = txn.get(key)
            sample = pickle.loads(data_bytes)
            return sample

    def close(self):
        self.env.close()


def split_and_save_to_lmdb(input_jsonl_file, train_lmdb_path, test_lmdb_path,
                           train_ratio=0.7, seed=42, max_samples=None):
    """
    Split JSONL dataset into train/test, then save each as LMDB after processing
    with PubTabNetJSONLDataset.
    """
    random.seed(seed)

    # create temp files
    tmp_train = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')
    tmp_test = tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8')

    # Stream input file and split
    with open(input_jsonl_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            if random.random() < train_ratio:
                tmp_train.write(line)
            else:
                tmp_test.write(line)

    tmp_train.close()
    tmp_test.close()

    # Function to save dataset to LMDB
    def save_dataset_lmdb(jsonl_file, lmdb_path):
        dataset = PubTabNetJSONLDataset(jsonl_file, max_samples=None)
        env = lmdb.open(lmdb_path, map_size=10**10)  # adjust size if needed
        sample_num = 0
        with env.begin(write=True) as txn:
            for idx, sample in enumerate(tqdm(dataset, desc=f"Saving {lmdb_path}")):
                key = f"{sample_num:08}".encode('utf-8')
                if sample['meta']['rows'] > 0 and sample['meta']['cols'] > 0:
                    value = pickle.dumps(sample)
                    txn.put(key, value)
                    sample_num += 1
        env.close()

    # save train/test datasets
    save_dataset_lmdb(tmp_train.name, train_lmdb_path)
    save_dataset_lmdb(tmp_test.name, test_lmdb_path)

    # cleanup temp files
    os.unlink(tmp_train.name)
    os.unlink(tmp_test.name)

    print(f"LMDB saved: train -> {train_lmdb_path}, test -> {test_lmdb_path}")


def visualize_table_data(table_data: Dict[str, Any], show_atom_grid: bool = True):
    """
    Visualizes a table based on the output of the modified PubTabNetJSONLDataset's generate method.

    Args:
        table_data (Dict[str, Any]): The output dictionary from the dataset's generate method.
        show_atom_grid (bool): Whether to draw the underlying atomic grid.
    """
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    cells = table_data.get("cells", [])
    meta = table_data.get("meta", {})
    atom_to_merged = table_data.get("atom_to_merged", [])

    # 1. Visualize merged cells
    for i, cell in enumerate(cells):
        x0, y0, x1, y1 = cell["cell_bbox"]

        # Use a random color for each merged cell for better distinction
        color = (random.random(), random.random(), random.random())

        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

        # Add cell info (row/col ID and span)
        ax.text(x0, y0 - 0.01,
                f"R{cell['row_id']}C{cell['col_id']}({cell['row_span']}x{cell['col_span']})",
                fontsize=8, color='black')

        # Visualize text bboxes inside the merged cell
        if cell["has_text"] and cell["text_bboxes"]:
            for t_bbox in cell["text_bboxes"]:
                tx0, ty0, tx1, ty1 = t_bbox
                text_rect = patches.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
                                              linewidth=0.5, edgecolor='gray', facecolor='gray', alpha=0.5)
                ax.add_patch(text_rect)

    # 2. Optionally visualize the underlying atomic grid
    if show_atom_grid and meta.get("rows") and meta.get("cols"):
        rows, cols = meta["rows"], meta["cols"]
        row_h = 1.0 / rows
        col_w = 1.0 / cols

        for r in range(rows):
            for c in range(cols):
                x0, y0 = c * col_w, r * row_h
                rect = patches.Rectangle((x0, y0), col_w, row_h,
                                         linewidth=0.5, edgecolor='lightgray', linestyle=':', facecolor='none')
                ax.add_patch(rect)

    plt.title("PubTabNet Sample (Modified Data Format)")
    plt.show()


def show_data():
    dataset = PubTabNetJSONLDataset('/media/win/Dataset/PubTabNet/PubTabNet_2.0.0.jsonl', max_samples=100)
    max_cols = max_rows = 0
    for i in range(len(dataset)):
        if i % 100 == 0:
            print('%d/%d ...' % (i, len(dataset)))
            print(max_rows, max_cols)
        sample_data = dataset.generate(i)
        visualize_table_data(sample_data)
        max_rows = max(max_rows, sample_data['meta']['rows'])
        max_cols = max(max_cols, sample_data['meta']['cols'])


if __name__ == "__main__":
    task = 'show_data'
    if task == 'show_data':
        show_data()
    elif task == 'split_and_save_to_lmdb':
        split_and_save_to_lmdb(input_jsonl_file='/media/win/Dataset/PubTabNet/PubTabNet_2.0.0.jsonl',
                               train_lmdb_path='/media/win/Dataset/PubTabNet/lmdb/train',
                               test_lmdb_path='/media/win/Dataset/PubTabNet/lmdb/test',
                               train_ratio=0.8)
