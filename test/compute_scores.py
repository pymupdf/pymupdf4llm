import json
import os
import sys
from pathlib import Path

from tabulate import tabulate


def compute_scores(filename):
    counts = json.loads(Path(filename).read_text())
    print()
    filename = counts.get("Header", os.path.basename(filename))
    title = f"Scores for {filename}"
    print(title)
    print("=" * len(title))
    table = [
        [
            "Category",
            "Precision",
            "Recall",
            "F1 Score",
            "GT Count",
            "Det Count",
            "TP",
            "FP",
            "FN",
        ]
    ]
    GP = GDT = GTP = GFP = GFN = 0

    for bclass in sorted(counts.keys()):
        if bclass == "Header":
            continue
        P = counts[bclass]["P"]
        TP = counts[bclass]["TP"]
        FP = counts[bclass]["FP"]
        FN = P - TP
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        t = [
            f"{bclass}",
            f"{precision:.4f}",
            f"{recall:.4f}",
            f"{f1:.4f}",
            f"{P}",
            f"{TP+FP}",
            f"{TP}",
            f"{FP}",
            f"{FN}",
        ]
        table.append(t)

        GP += P
        GTP += TP
        GFP += FP
        GFN += FN
    Gprecision = GTP / (GTP + GFP) if (GTP + GFP) > 0 else 0
    Grecall = GTP / (GTP + GFN) if (GTP + GFN) > 0 else 0
    GF1 = (
        2 * Gprecision * Grecall / (Gprecision + Grecall)
        if (Gprecision + Grecall) > 0
        else 0
    )
    t = [
        "OVERALL",
        f"{Gprecision:.4f}",
        f"{Grecall:.4f}",
        f"{GF1:.4f}",
        f"{GP}",
        f"{GTP+GFP}",
        f"{GTP}",
        f"{GFP}",
        f"{GFN}",
    ]
    table.append(t)
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


if __name__ == "__main__":
    filename = sys.argv[1]
    compute_scores(filename)
