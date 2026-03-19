"""Shortcut-proof synthetic dataset for spatial measurement evaluation.

Generates technical drawings of a single hole with a scale bar. The only
valid path to the correct diameter is the scale bar formula:

    diameter_mm = hole_pixels × (scale_bar_mm / scale_bar_pixels)

All other correlations (pixel diameter, canvas size, zoom level) are
decorrelated by construction. Verified statistically at generation time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
import numpy as np
from numpy.random import Generator

SCALE_BAR_VALUES: list[int] = [5, 10, 15, 20, 25, 30, 40, 50]
DIAMETER_RANGE: tuple[float, float] = (3.0, 30.0)
CANVAS_W_RANGE: tuple[int, int] = (500, 1000)
CANVAS_H_RANGE: tuple[int, int] = (400, 800)
DPI: int = 150

PLATE_COLOR: str = "#F0F0EA"
PLATE_EDGE_COLOR: str = "black"
HOLE_EDGE_COLOR: str = "black"


def compute_ppm_range(
    diameter_mm: float,
    scale_bar_mm: int,
    canvas_w: int,
    canvas_h: int,
) -> tuple[float, float]:
    """Return feasible (min, max) pixels-per-mm so hole and scale bar fit the canvas."""
    min_hole_px = 15
    min_sb_px = 30
    max_feature_frac = 0.55

    ppm_min = max(min_hole_px / diameter_mm, min_sb_px / scale_bar_mm)
    max_feature_mm = max(diameter_mm, scale_bar_mm)
    ppm_max = max_feature_frac * canvas_w / max_feature_mm
    ppm_max = min(ppm_max, 0.5 * canvas_h / diameter_mm)

    return ppm_min, ppm_max


def generate_sample(rng: Generator, idx: int) -> dict[str, Any]:
    """Generate metadata for one training/test image with independent random parameters."""
    diameter_mm = rng.uniform(*DIAMETER_RANGE)
    scale_bar_mm = int(rng.choice(SCALE_BAR_VALUES))
    canvas_w = int(rng.integers(*CANVAS_W_RANGE))
    canvas_h = int(rng.integers(*CANVAS_H_RANGE))

    ppm_min, ppm_max = compute_ppm_range(diameter_mm, scale_bar_mm, canvas_w, canvas_h)

    if ppm_min >= ppm_max:
        canvas_w = int(canvas_w * 1.5)
        canvas_h = int(canvas_h * 1.5)
        ppm_min, ppm_max = compute_ppm_range(diameter_mm, scale_bar_mm, canvas_w, canvas_h)

    ppm = rng.uniform(ppm_min, ppm_max)
    hole_px = diameter_mm * ppm
    sb_px = scale_bar_mm * ppm

    plate_w_min = max(hole_px * 1.8, 100)
    plate_w_max = max(plate_w_min + 50, canvas_w * 0.75)
    plate_w_px = rng.uniform(plate_w_min, plate_w_max)

    plate_h_min = max(hole_px * 1.8, 80)
    plate_h_max = max(plate_h_min + 50, canvas_h * 0.65)
    plate_h_px = rng.uniform(plate_h_min, plate_h_max)

    plate_x_max = max(21, canvas_w - plate_w_px - 20)
    plate_x = rng.uniform(20, plate_x_max)
    plate_y_max = max(61, canvas_h - plate_h_px - 20)
    plate_y = rng.uniform(60, plate_y_max)

    margin = hole_px / 2 + 10
    hole_cx = rng.uniform(plate_x + margin, max(plate_x + margin + 1, plate_x + plate_w_px - margin))
    hole_cy = rng.uniform(plate_y + margin, max(plate_y + margin + 1, plate_y + plate_h_px - margin))

    sb_x = plate_x
    sb_y = plate_y - 35

    return {
        "idx": idx,
        "diameter_mm": round(float(diameter_mm), 2),
        "scale_bar_mm": scale_bar_mm,
        "canvas_w": canvas_w,
        "canvas_h": canvas_h,
        "ppm": round(float(ppm), 4),
        "hole_px": round(float(hole_px), 1),
        "sb_px": round(float(sb_px), 1),
        "plate_x": float(plate_x),
        "plate_y": float(plate_y),
        "plate_w_px": float(plate_w_px),
        "plate_h_px": float(plate_h_px),
        "hole_cx": float(hole_cx),
        "hole_cy": float(hole_cy),
        "sb_x": float(sb_x),
        "sb_y": float(sb_y),
    }


def render_image(sample: dict[str, Any], output_path: str) -> None:
    """Render a technical drawing with one hole and a labeled scale bar."""
    cw, ch = sample["canvas_w"], sample["canvas_h"]
    fig, ax = plt.subplots(1, 1, figsize=(cw / DPI, ch / DPI), dpi=DPI)
    ax.set_xlim(0, cw)
    ax.set_ylim(0, ch)
    ax.set_aspect("equal")
    ax.axis("off")

    rect = mpl_patches.Rectangle(
        (sample["plate_x"], sample["plate_y"]),
        sample["plate_w_px"], sample["plate_h_px"],
        linewidth=1.5, edgecolor=PLATE_EDGE_COLOR,
        facecolor=PLATE_COLOR, zorder=1,
    )
    ax.add_patch(rect)

    hole_r = sample["hole_px"] / 2
    circle = mpl_patches.Circle(
        (sample["hole_cx"], sample["hole_cy"]),
        hole_r, linewidth=1.2,
        edgecolor=HOLE_EDGE_COLOR, facecolor="white", zorder=2,
    )
    ax.add_patch(circle)

    cx, cy = sample["hole_cx"], sample["hole_cy"]
    cross_len = hole_r * 1.3
    ax.plot([cx - cross_len, cx + cross_len], [cy, cy], color="#999999", linewidth=0.7, zorder=3)
    ax.plot([cx, cx], [cy - cross_len, cy + cross_len], color="#999999", linewidth=0.7, zorder=3)
    ax.text(cx, cy + hole_r + 8, "H1", ha="center", va="bottom", fontsize=7, fontweight="bold", zorder=4)

    sb_x, sb_y, sb_len = sample["sb_x"], sample["sb_y"], sample["sb_px"]
    ax.plot([sb_x, sb_x + sb_len], [sb_y, sb_y], color="black", linewidth=2, zorder=5)
    tick_h = 5
    ax.plot([sb_x, sb_x], [sb_y - tick_h, sb_y + tick_h], color="black", linewidth=1.5, zorder=5)
    ax.plot([sb_x + sb_len, sb_x + sb_len], [sb_y - tick_h, sb_y + tick_h], color="black", linewidth=1.5, zorder=5)
    ax.text(sb_x + sb_len / 2, sb_y - 10, f"{sample['scale_bar_mm']} mm", ha="center", va="top", fontsize=7, zorder=5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def verify_no_shortcuts(samples: list[dict[str, Any]]) -> None:
    """Print correlation analysis proving no single feature predicts diameter."""
    diams = np.array([s["diameter_mm"] for s in samples])
    hole_pxs = np.array([s["hole_px"] for s in samples])
    sb_mms = np.array([s["scale_bar_mm"] for s in samples])
    sb_pxs = np.array([s["sb_px"] for s in samples])
    canvas_ws = np.array([s["canvas_w"] for s in samples])
    ppms = np.array([s["ppm"] for s in samples])
    correct = hole_pxs * sb_mms / sb_pxs

    print("=== SHORTCUT VERIFICATION ===")
    print(f"N = {len(samples)}")
    print(f"Diameter range: {diams.min():.1f} - {diams.max():.1f} mm")
    print(f"Diameter mean: {diams.mean():.1f} mm\n")

    features = {
        "hole_pixels": hole_pxs,
        "scale_bar_mm": sb_mms,
        "scale_bar_pixels": sb_pxs,
        "canvas_width": canvas_ws,
        "pixels_per_mm": ppms,
        "hole_px / canvas_w": hole_pxs / canvas_ws,
        "CORRECT (hole_px * sb_mm / sb_px)": correct,
    }

    print("Correlation with diameter_mm:")
    for name, feat in features.items():
        r = np.corrcoef(diams, feat)[0, 1]
        print(f"  {name:<45s} r = {r:+.3f}")
    print()

    from numpy.polynomial import polynomial as P

    mean_mae = np.mean(np.abs(diams - diams.mean()))
    median_mae = np.mean(np.abs(diams - np.median(diams)))
    c = P.polyfit(hole_pxs, diams, 1)
    lr_mae = np.mean(np.abs(diams - P.polyval(hole_pxs, c)))
    correct_mae = np.mean(np.abs(diams - correct))

    print("Baseline MAEs (no vision needed):")
    print(f"  Always guess mean ({diams.mean():.1f}mm):     {mean_mae:.2f} mm")
    print(f"  Always guess median ({np.median(diams):.1f}mm):   {median_mae:.2f} mm")
    print(f"  Linear regress on pixel diameter:  {lr_mae:.2f} mm")
    print(f"  Correct formula (scale bar):       {correct_mae:.4f} mm\n")


def generate_matched_pairs(rng: Generator, n_pairs: int = 50) -> list[dict[str, Any]]:
    """Generate image pairs with identical pixel holes but different scale bars.

    Each pair shares the same hole pixel diameter, canvas size, and hole position.
    The scale bar differs in both mm label and pixel length, producing different
    ground truth diameters. This is the causal test for scale bar usage.
    """
    pairs: list[dict[str, Any]] = []
    for i in range(n_pairs):
        sb_pair = rng.choice(SCALE_BAR_VALUES, size=2, replace=False)
        sb_a, sb_b = int(sb_pair[0]), int(sb_pair[1])

        canvas_w = int(rng.integers(600, 900))
        canvas_h = int(rng.integers(500, 700))
        hole_px = float(rng.uniform(30, 200))

        ppm_a = float(rng.uniform(3.0, 15.0))
        ppm_b = float(rng.uniform(3.0, 15.0))
        while abs(ppm_a - ppm_b) < 1.5:
            ppm_b = float(rng.uniform(3.0, 15.0))

        diam_a = hole_px / ppm_a
        diam_b = hole_px / ppm_b
        sb_px_a = sb_a * ppm_a
        sb_px_b = sb_b * ppm_b

        max_feature = max(hole_px, sb_px_a, sb_px_b)
        if max_feature > canvas_w * 0.6 or hole_px > canvas_h * 0.4:
            continue

        plate_w = max(hole_px * 2, 150)
        plate_h = max(hole_px * 2, 120)
        plate_x = (canvas_w - plate_w) / 2
        plate_y = 70.0
        hole_cx = canvas_w / 2
        hole_cy = plate_y + plate_h / 2

        base = {
            "canvas_w": canvas_w, "canvas_h": canvas_h,
            "hole_px": round(hole_px, 1),
            "plate_x": plate_x, "plate_y": plate_y,
            "plate_w_px": plate_w, "plate_h_px": plate_h,
            "hole_cx": hole_cx, "hole_cy": hole_cy,
            "sb_x": plate_x, "sb_y": plate_y - 35,
        }

        var_a = {**base, "idx": i, "diameter_mm": round(diam_a, 2), "scale_bar_mm": sb_a, "ppm": round(ppm_a, 4), "sb_px": round(sb_px_a, 1)}
        var_b = {**base, "idx": i, "diameter_mm": round(diam_b, 2), "scale_bar_mm": sb_b, "ppm": round(ppm_b, 4), "sb_px": round(sb_px_b, 1)}

        pairs.append({
            "pair_id": i, "hole_px": round(hole_px, 1),
            "a": var_a, "b": var_b,
            "diam_a": round(diam_a, 2), "diam_b": round(diam_b, 2),
            "sb_a": sb_a, "sb_b": sb_b,
        })

    return pairs


def verify_matched_pairs(pairs: list[dict[str, Any]]) -> None:
    """Print matched pair statistics for the diagnostic set."""
    diffs = [abs(p["diam_a"] - p["diam_b"]) for p in pairs]
    print("\n=== MATCHED PAIR DIAGNOSTIC ===")
    print(f"N pairs: {len(pairs)}")
    print(f"Ground truth difference per pair: {np.mean(diffs):.1f} mm avg (range {np.min(diffs):.1f} - {np.max(diffs):.1f})")
    print("Same hole pixels, different scale bars -> different correct answers")
    print("If model gives same answer for both: NOT using scale bar")
    print("If model answers diverge correctly: USING scale bar\n")
    for p in pairs[:5]:
        print(f"  Pair {p['pair_id']}: hole={p['hole_px']}px, "
              f"SB {p['sb_a']}mm->{p['diam_a']}mm vs SB {p['sb_b']}mm->{p['diam_b']}mm "
              f"(diff={abs(p['diam_a'] - p['diam_b']):.1f}mm)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shortcut-proof spatial measurement dataset")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="dataset")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    train_samples = [generate_sample(rng, i) for i in range(args.n_train)]
    test_samples = [generate_sample(rng, i) for i in range(args.n_test)]

    print("=== TRAIN SET ===")
    verify_no_shortcuts(train_samples)
    print("=== TEST SET ===")
    verify_no_shortcuts(test_samples)

    if args.verify_only:
        return

    out = Path(args.output_dir)
    for split in ["train", "test", "test_matched"]:
        (out / split).mkdir(parents=True, exist_ok=True)

    for split, samples in [("train", train_samples), ("test", test_samples)]:
        with open(out / split / "metadata.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        for i, s in enumerate(samples):
            render_image(s, str(out / split / f"image_{s['idx']:04d}.png"))
            if (i + 1) % 100 == 0:
                print(f"  {split}: {i + 1}/{len(samples)}")

    print("\nGenerating matched pair diagnostic set...")
    matched_pairs = generate_matched_pairs(rng, n_pairs=50)

    with open(out / "test_matched" / "metadata.jsonl", "w") as f:
        for pair in matched_pairs:
            f.write(json.dumps(pair) + "\n")

    for pair in matched_pairs:
        for variant in ["a", "b"]:
            render_image(pair[variant], str(out / "test_matched" / f"pair_{pair['pair_id']:03d}_{variant}.png"))

    print(f"  Generated {len(matched_pairs)} matched pairs")
    verify_matched_pairs(matched_pairs)

    print(f"\nDone. Output in {out}/")
    print(f"  train/: {len(train_samples)} images")
    print(f"  test/: {len(test_samples)} images")
    print(f"  test_matched/: {len(matched_pairs)} pairs ({len(matched_pairs) * 2} images)")


if __name__ == "__main__":
    main()
