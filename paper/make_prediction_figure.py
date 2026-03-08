"""Generate qualitative prediction figure for the paper.

Runs both trained models (GoogLeNet and MobileNetV2) on a few sample images
and saves a multi-panel figure under paper/figures/.

Usage (from repo root):
  D:/skin_disease_project/venv/Scripts/python.exe paper/make_prediction_figure.py

This script is intentionally deterministic and does not require GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
TABLE_TEX = ROOT / "paper" / "predictions_table.tex"
ACC_TEX = ROOT / "paper" / "predictions_accuracy.tex"

GOOGLENET_CKPT = ROOT / "GoogLeNet_best.pth"
MOBILENET_CKPT = ROOT / "MobileNetV2_best.pth"

def get_class_names() -> list[str]:
    """Return class names in the same order ImageFolder uses (sorted directory names)."""
    train_dir = ROOT / "dataset" / "train_set"
    if train_dir.exists():
        return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    # Fallback to the known class set (kept for robustness)
    return [
        "BA- cellulitis",
        "BA-impetigo",
        "FU-athlete-foot",
        "FU-nail-fungus",
        "FU-ringworm",
        "PA-cutaneous-larva-migrans",
        "VI-chickenpox",
        "VI-shingles",
    ]


@dataclass
class Pred:
    label: str
    confidence: float


def create_googlenet(num_classes: int) -> torch.nn.Module:
    model = models.googlenet(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model.eval()
    return model


def create_mobilenetv2(num_classes: int) -> torch.nn.Module:
    model = models.mobilenet_v2(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model.eval()
    return model


def load_checkpoint(model: torch.nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)


def preprocess(image_path: Path) -> torch.Tensor:
    tfm = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(image_path).convert("RGB")
    return tfm(img).unsqueeze(0)


def predict_top1(model: torch.nn.Module, x: torch.Tensor) -> Pred:
    with torch.no_grad():
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1)[0]
        conf, idx = torch.max(probs, dim=0)
    class_names = get_class_names()
    return Pred(label=class_names[int(idx)], confidence=float(conf))


def read_labels_csv(labels_path: Path) -> dict[str, str]:
    """Read filename,true_label from a CSV (header optional)."""
    mapping: dict[str, str] = {}
    if not labels_path.exists():
        return mapping
    with labels_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return mapping
    # Handle optional header
    start = 0
    header = [c.strip().lower() for c in rows[0]]
    if len(header) >= 2 and ("file" in header[0] or "name" in header[0]) and ("label" in header[1] or "class" in header[1]):
        start = 1
    for r in rows[start:]:
        if len(r) < 2:
            continue
        mapping[r[0].strip()] = r[1].strip()
    return mapping


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    class_names = get_class_names()

    new_test_dir = ROOT / "dataset" / "new_random_test"
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    candidates = []
    if new_test_dir.exists():
        candidates = [p for p in sorted(new_test_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    used_new_random = len(candidates) >= 4
    if used_new_random:
        samples = candidates[:4]
    else:
        print(
            "WARNING: dataset/new_random_test has fewer than 4 images. "
            "Falling back to demo images from dataset/test_set."
        )
        samples = [
            ROOT / "dataset" / "test_set" / "VI-chickenpox" / "0_VI-chickenpox (2).jpeg",
            ROOT / "dataset" / "test_set" / "FU-ringworm" / "62_FU-ringworm (8).jpeg",
            ROOT / "dataset" / "test_set" / "BA-impetigo" / "42_BA-impetigo (2).png",
            ROOT / "dataset" / "test_set" / "VI-shingles" / "27_VI-shingles (1).jpg",
        ]
        samples = [p for p in samples if p.exists()]
        if len(samples) < 2:
            raise RuntimeError("Not enough demo images found in dataset/test_set.")

    goog = create_googlenet(len(class_names))
    mob = create_mobilenetv2(len(class_names))
    load_checkpoint(goog, GOOGLENET_CKPT)
    load_checkpoint(mob, MOBILENET_CKPT)

    labels_map = read_labels_csv(new_test_dir / "labels.csv") if used_new_random else {}

    rows = len(samples)
    fig, axes = plt.subplots(rows, 1, figsize=(7.2, 2.2 * rows), dpi=200)
    if rows == 1:
        axes = [axes]

    preds_for_table: list[tuple[str, str | None, Pred, Pred]] = []

    for ax, img_path in zip(axes, samples):
        img = Image.open(img_path).convert("RGB")
        x = preprocess(img_path)

        p_g = predict_top1(goog, x)
        p_m = predict_top1(mob, x)

        ax.imshow(img)
        ax.axis("off")
        origin = "new_random_test" if used_new_random else img_path.parent.name
        ax.set_title(
            f"Image: {origin}/{img_path.name} | GoogLeNet: {p_g.label} ({p_g.confidence*100:.1f}%) | "
            f"MobileNetV2: {p_m.label} ({p_m.confidence*100:.1f}%)",
            fontsize=8,
        )

        true_label = labels_map.get(img_path.name)
        preds_for_table.append((img_path.name, true_label, p_g, p_m))

    out_path = FIG_DIR / "prediction_examples.png"
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    # Write a LaTeX table snippet for direct \input in the paper
    caption_src = (
        "Random test images from dataset/new\\_random\\_test." if used_new_random else "Demo images from the held-out test set."
    )
    lines: list[str] = []
    lines.append("% Auto-generated by paper/make_prediction_figure.py")
    lines.append("\\begin{table}[ht]")
    lines.append("\\caption{Manual prediction examples (top-1) using both models. " + caption_src + "}")
    lines.append("\\label{tab:qual}")
    lines.append("\\centering")
    lines.append("\\small")
    has_gt = any(t is not None and t != "" for _, t, _, _ in preds_for_table)
    lines.append("\\resizebox{\\columnwidth}{!}{%")
    lines.append("\\begin{tabular}{l" + ("l" if has_gt else "") + "cc}")
    lines.append("\\toprule")
    if has_gt:
        lines.append("\\textbf{Image} & \\textbf{True label} & \\textbf{GoogLeNet} & \\textbf{MobileNetV2}\\\\")
    else:
        lines.append("\\textbf{Image} & \\textbf{GoogLeNet} & \\textbf{MobileNetV2}\\\\")
    lines.append("\\midrule")
    correct_g = 0
    correct_m = 0
    total_gt = 0

    for name, true_label, pg, pm in preds_for_table:
        # Escape underscores for LaTeX
        safe_name = name.replace("_", "\\_")
        if true_label and true_label.strip():
            total_gt += 1
            if pg.label == true_label:
                correct_g += 1
            if pm.label == true_label:
                correct_m += 1
            safe_true = true_label.replace("_", "\\_")
            lines.append(
                f"{safe_name} & {safe_true} & {pg.label} ({pg.confidence*100:.1f}\\%) & {pm.label} ({pm.confidence*100:.1f}\\%)\\\\"
            )
        else:
            lines.append(
                f"{safe_name} & {pg.label} ({pg.confidence*100:.1f}\\%) & {pm.label} ({pm.confidence*100:.1f}\\%)\\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    TABLE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved: {TABLE_TEX}")

    # Write a short accuracy paragraph if ground-truth labels were provided
    if total_gt > 0:
        acc_g = 100.0 * correct_g / total_gt
        acc_m = 100.0 * correct_m / total_gt
        acc_text = (
            "\\noindent "
            f"On the 4 random test images with provided ground-truth labels, GoogLeNet achieved "
            f"{acc_g:.1f}\\% accuracy and MobileNetV2 achieved {acc_m:.1f}\\% accuracy.\\\\\n"
        )
    else:
        acc_text = "% No ground-truth labels.csv provided for dataset/new_random_test\n"
    ACC_TEX.write_text(acc_text, encoding="utf-8")
    print(f"Saved: {ACC_TEX}")

    # Also print a small table-like summary for manual copy
    print("\nPredictions summary:")
    for img_path in samples:
        x = preprocess(img_path)
        p_g = predict_top1(goog, x)
        p_m = predict_top1(mob, x)
        print(f"- {img_path.name}: GoogLeNet={p_g.label} ({p_g.confidence*100:.1f}%), MobileNetV2={p_m.label} ({p_m.confidence*100:.1f}%)")


if __name__ == "__main__":
    main()
