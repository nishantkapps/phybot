#!/usr/bin/env python3
"""
1-NN classification and Silhouette analysis on ground-truth pose data.
Assesses exercise correctness by clustering sequences and evaluating nearest-neighbour labels.

Usage:
  python nn_exercise_quality.py --mode 3d --examples Ex1 Ex2 --label_map labels.json

Inputs:
  - Pose data under ./2d_joints/Ex* or ./3d_joints/Ex*
  - labels.json: { "Ex1": "label_name", "Ex2": "label_name", ... }

Outputs:
  - Prints 1-NN leave-one-out accuracy
  - Prints silhouette coefficient per label
  - Shows PCA scatter with class colors and a silhouette plot
"""

import argparse
import json
import os
import csv
import pandas as pd
import re
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt

from model3 import load_pose_sequence
from DataLoader import load_video_frames


def extract_features(pose_seq: np.ndarray) -> np.ndarray:
    """
    Compute simple, robust features per sequence: 
    mean and std of flattened joints over time.
    Returns a 1D feature vector.
    """
    # pose_seq: [frames, joints, dims]
    frames = pose_seq.shape[0]
    flat = pose_seq.reshape(frames, -1)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return np.concatenate([mean, std], axis=0)


def load_dataset_single_example(data_dir: str, mode: str, example: str, max_frames: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Load ALL sequences for a single example folder.
    Returns features matrix X, default labels (example repeated), 
    and file identifiers (example/stem).
    """
    X: List[np.ndarray] = []
    y: List[str] = []
    ids: List[str] = []
    folder = os.path.join(data_dir, ("2d_joints" if mode == "2d" else "3d_joints"), example)
    vframes, ffiles = load_video_frames(folder)
    if not vframes:
        raise RuntimeError(f"No sequences found in {folder}")
    for i, seq in enumerate(vframes):
        try:
            seq_clip = seq[:max_frames]
            feats = extract_features(seq_clip)
            X.append(feats)
            y.append(example)
            stem = os.path.splitext(ffiles[i])[0] if i < len(ffiles) else str(i)
            ids.append(f"{example}/{stem}")
        except Exception as e:
            print(f"[warn] failed to process {example} file {i}: {e}")
            continue
    return np.vstack(X), y, ids


def _normalize_stem(name: str) -> str:
    s = os.path.splitext(name.strip())[0]
    s = s.strip().lower()
    # Strip common camera/framerate suffixes like -c17-120fps or -c18-30fps
    s = re.sub(r"-c\d{2}-\d+fps$", "", s)
    # Also strip plain framerate suffix like -30fps
    s = re.sub(r"-\d+fps$", "", s)
    return s

def _first_six(stem: str) -> str:
    return stem.strip().lower()[:6]


def load_correctness_labels_first6(seg_csv_path: str) -> Dict[str, int]:
    """Load binary correctness labels from Segmentation.csv using video_id first 6 chars.
    Returns mapping from first6 -> 0/1.
    """
    if not os.path.isfile(seg_csv_path):
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv_path}")
    # Robust CSV read with automatic delimiter detection
    df = pd.read_csv(seg_csv_path, engine="python", sep=None)
    cols = {c.lower(): c for c in df.columns}
    vid_col = cols.get("video_id")
    lab_col = cols.get("correctness")
    if vid_col is None or lab_col is None:
        raise RuntimeError("Segmentation.csv must contain columns 'video_id' and 'correctness'")
    vids = df[vid_col].astype(str).str.strip().str.lower().str[:6]
    corr = df[lab_col].astype(float).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)
    return {vid: int(val) for vid, val in zip(vids, corr)}


def leave_one_out_1nn(*args, **kwargs):
    raise NotImplementedError


def plot_pca_scatter(*args, **kwargs):
    raise NotImplementedError


def plot_silhouette(*args, **kwargs):
    raise NotImplementedError


def plot_correctness_by_file(ids: List[str], y_bin: List[int], title: str) -> None:
    """Plot a simple chart showing correctness (0/1) for each example/file id."""
    if not ids:
        return
    # Sort by example then filename for readability
    items = sorted(zip(ids, y_bin), key=lambda t: t[0].lower())
    # Keep only 30fps files
    items = [it for it in items if "-30fps" in it[0].lower()]
    labels = [it[0] for it in items]
    vals = [int(it[1]) for it in items]
    y_pos = np.arange(len(labels))
    colors = ['#2ca02c' if v == 1 else '#d62728' for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3*len(labels))))
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('Correctness (0=incorrect, 1=correct)')
    ax.set_title(title)
    # Annotate bars
    for i, v in enumerate(vals):
        ax.text(v + 0.02 if v == 1 else v + 0.02, i + 0.1, str(v), color='black', fontsize=8)
    plt.tight_layout()


def plot_repetition_counts_ax(ax, example: str, ffiles: List[str], seg_csv_path: str, title: str) -> None:
    """Plot per-file repetition counts (correct vs incorrect) for all files in example.

    - Every .npy file in the example is included (even if counts are zero).
    - Uses Segmentation.csv columns: video_id, correctness
    """
    import pandas as pd
    if not os.path.isfile(seg_csv_path):
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv_path}")
    df = pd.read_csv(seg_csv_path, engine="python", sep=None)
    cols = {c.lower(): c for c in df.columns}
    vid_col = cols.get("video_id")
    lab_col = cols.get("correctness")
    if vid_col is None or lab_col is None:
        raise RuntimeError("Segmentation.csv must contain columns 'video_id' and 'correctness'")
    # Normalize video_id to first-6 (lower)
    df["vid6"] = df[vid_col].astype(str).str.strip().str.lower().str[:6]
    df["corr"] = df[lab_col].astype(float).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)

    # Build list of stems for files in this example
    stems = [os.path.splitext(fn)[0] for fn in ffiles]
    # Include both 30fps and 120fps variants
    vid6_list = [s.strip().lower()[:6] for s in stems]

    # Sort stems by numeric ID in pattern PM_### if present
    def sort_key(stem: str) -> int:
        m = re.search(r"pm[_-]?(\d+)", stem.lower())
        return int(m.group(1)) if m else 10**9
    order = sorted(range(len(stems)), key=lambda i: sort_key(stems[i]))
    stems = [stems[i] for i in order]
    vid6_list = [vid6_list[i] for i in order]

    # Count per vid6
    counts = df.groupby("vid6")["corr"].value_counts().unstack(fill_value=0)
    # Ensure columns 0,1 exist
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts[[0, 1]]

    # Prepare plotting data in file order
    incorrect = []
    correct = []
    labels = []
    for stem, v6 in zip(stems, vid6_list):
        labels.append(f"{example}/{stem}")
        if v6 in counts.index:
            incorrect.append(int(counts.loc[v6, 0]))
            correct.append(int(counts.loc[v6, 1]))
        else:
            incorrect.append(0)
            correct.append(0)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, incorrect, color="#d62728", alpha=0.8, label="Incorrect")
    ax.barh(y_pos, correct, left=incorrect, color="#2ca02c", alpha=0.8, label="Correct")
    # Shorten labels to PM_###-XXfps and thin out tick labels to reduce clutter
    short_labels = []
    for lab in labels:
        # lab format: Ex*/<stem>; extract example and PM_### id
        try:
            ex_part, stem_part = lab.split('/', 1)
        except ValueError:
            ex_part, stem_part = example, lab
        m = re.search(r"(PM[_-]?\d+)", stem_part, flags=re.IGNORECASE)
        pm_id = m.group(1).upper() if m else stem_part
        short_labels.append(f"{ex_part}/{pm_id}")
    thin = max(1, len(short_labels)//20)
    shown = [lbl if (i % thin == 0) else "" for i, lbl in enumerate(short_labels)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shown, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Repetition count")
    ax.set_title(title)
    # Show legend only if there is space; caller may hide for other panels
    ax.legend(loc='best', fontsize=8)
    # Annotate incorrect and correct counts per bar (no percentage)
    totals = [i + c for i, c in zip(incorrect, correct)]
    for i, (inc, cor, tot) in enumerate(zip(incorrect, correct, totals)):
        # Incorrect count on left segment (if nonzero)
        if inc > 0:
            ax.text(max(inc * 0.5, 0.05), i, str(inc), va="center", ha="center", fontsize=8, color="#ffffff")
        # Correct count on green segment (if nonzero)
        if cor > 0:
            ax.text(inc + max(cor * 0.5, 0.05), i, str(cor), va="center", ha="center", fontsize=8, color="#ffffff")
    ax.figure.tight_layout()


def main():
    parser = argparse.ArgumentParser(description="1-NN classification and silhouette analysis for exercise correctness")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], default="3d")
    parser.add_argument("--examples", nargs='*', help="Example folders (e.g., Ex1 Ex2 ...). If omitted, autodetect Ex* folders")
    parser.add_argument("--segmentation_csv", type=str, default="Segmentation.csv", help="CSV with ground-truth correctness labels")
    parser.add_argument("--max_frames", type=int, default=300)
    args = parser.parse_args()
    # Build example list
    examples = args.examples if args.examples else []
    if not examples:
        base = os.path.join(args.data_dir, "2d_joints" if args.mode == "2d" else "3d_joints")
        examples = sorted([d for d in os.listdir(base) if d.lower().startswith("ex") and os.path.isdir(os.path.join(base, d))])
        if not examples:
            raise RuntimeError(f"No Ex* folders found in {base}")

    # Create a grid for up to 6 examples (2 rows x 3 cols)
    n = len(examples)
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(18, max(6, 3*n/cols)))
    axes = axes.flatten()

    for idx, ex in enumerate(examples[:rows*cols]):
        ax = axes[idx]
        folder = os.path.join(args.data_dir, ("2d_joints" if args.mode == "2d" else "3d_joints"), ex)
        _, ffiles = load_video_frames(folder)
        plot_repetition_counts_ax(ax, ex, ffiles, args.segmentation_csv, title=f"{ex} - Repetition counts")

    # Hide any unused axes
    for j in range(len(examples), rows*cols):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
1-NN classification and Silhouette analysis on ground-truth pose data.
Assesses exercise correctness by clustering sequences and evaluating nearest-neighbour labels.

Usage:
  python nn_exercise_quality.py --mode 3d --examples Ex1 Ex2 --label_map labels.json

Inputs:
  - Pose data under ./2d_joints/Ex* or ./3d_joints/Ex*
  - labels.json: { "Ex1": "label_name", "Ex2": "label_name", ... }

Outputs:
  - Prints 1-NN leave-one-out accuracy
  - Prints silhouette coefficient per label
  - Shows PCA scatter with class colors and a silhouette plot
"""

import argparse
import json
import os
import csv
import pandas as pd
import re
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score

from model3 import load_pose_sequence
from DataLoader import load_video_frames


def extract_features(pose_seq: np.ndarray) -> np.ndarray:
    """Compute simple, robust features per sequence: mean and std of flattened joints over time.
    Returns a 1D feature vector.
    """
    # pose_seq: [frames, joints, dims]
    frames = pose_seq.shape[0]
    flat = pose_seq.reshape(frames, -1)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return np.concatenate([mean, std], axis=0)


def load_dataset_single_example(data_dir: str, mode: str, example: str, max_frames: int) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load ALL sequences for a single example folder.
    Returns features matrix X, default labels (example repeated), and file identifiers (example/stem).
    """
    X: List[np.ndarray] = []
    y: List[str] = []
    ids: List[str] = []
    folder = os.path.join(data_dir, ("2d_joints" if mode == "2d" else "3d_joints"), example)
    vframes, ffiles = load_video_frames(folder)
    if not vframes:
        raise RuntimeError(f"No sequences found in {folder}")
    for i, seq in enumerate(vframes):
        try:
            seq_clip = seq[:max_frames]
            feats = extract_features(seq_clip)
            X.append(feats)
            y.append(example)
            stem = os.path.splitext(ffiles[i])[0] if i < len(ffiles) else str(i)
            ids.append(f"{example}/{stem}")
        except Exception as e:
            print(f"[warn] failed to process {example} file {i}: {e}")
            continue
    return np.vstack(X), y, ids


def _normalize_stem(name: str) -> str:
    s = os.path.splitext(name.strip())[0]
    s = s.strip().lower()
    # Strip common camera/framerate suffixes like -c17-120fps or -c18-30fps
    s = re.sub(r"-c\d{2}-\d+fps$", "", s)
    # Also strip plain framerate suffix like -30fps
    s = re.sub(r"-\d+fps$", "", s)
    return s

def _first_six(stem: str) -> str:
    return stem.strip().lower()[:6]


def load_correctness_labels_first6(seg_csv_path: str) -> Dict[str, int]:
    """Load binary correctness labels from Segmentation.csv using video_id first 6 chars.
    Returns mapping from first6 -> 0/1.
    """
    if not os.path.isfile(seg_csv_path):
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv_path}")
    # Robust CSV read with automatic delimiter detection
    df = pd.read_csv(seg_csv_path, engine="python", sep=None)
    cols = {c.lower(): c for c in df.columns}
    vid_col = cols.get("video_id")
    lab_col = cols.get("correctness")
    if vid_col is None or lab_col is None:
        raise RuntimeError("Segmentation.csv must contain columns 'video_id' and 'correctness'")
    vids = df[vid_col].astype(str).str.strip().str.lower().str[:6]
    corr = df[lab_col].astype(float).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)
    return {vid: int(val) for vid, val in zip(vids, corr)}


def leave_one_out_1nn(X: np.ndarray, y: List[str]) -> float:
    """Compute leave-one-out 1-NN accuracy using Euclidean distance on standardized features."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    n = Xs.shape[0]
    correct = 0
    for i in range(n):
        xi = Xs[i]
        dist = np.linalg.norm(Xs - xi, axis=1)
        dist[i] = np.inf
        j = int(np.argmin(dist))
        if y[j] == y[i]:
            correct += 1
    return correct / n


def plot_pca_scatter(X: np.ndarray, y: List[str], title: str) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)
    labels = np.unique(y)
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, lab in zip(colors, labels):
        mask = np.array([yi == lab for yi in y])
        ax.scatter(Z[mask, 0], Z[mask, 1], label=lab, color=c, alpha=0.8, edgecolors='k')
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_silhouette(X: np.ndarray, y: List[str], title: str) -> None:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    # Map string labels to ints for silhouette
    labels_unique = np.unique(y)
    y_int = np.array([int(np.where(labels_unique == yi)[0][0]) for yi in y])
    n_samples = Xs.shape[0]
    n_labels = len(labels_unique)
    # Guard: sklearn silhouette requires 2 <= n_labels <= n_samples-1
    if n_labels < 2 or n_labels > max(1, n_samples - 1):
        print(f"[warn] Silhouette requires 2..(n_samples-1) clusters; got n_labels={n_labels}, n_samples={n_samples}. Skipping plot.")
        # Also print label counts to help user fix labels
        vals, counts = np.unique(y, return_counts=True)
        print("[info] Label counts:", {v: int(c) for v, c in zip(vals, counts)})
        return
    sil_avg = silhouette_score(Xs, y_int, metric='euclidean')
    sil_vals = silhouette_samples(Xs, y_int, metric='euclidean')

    fig, ax = plt.subplots(figsize=(6, 5))
    y_lower = 10
    for i, lab in enumerate(labels_unique):
        ith_sil = sil_vals[y_int == i]
        ith_sil.sort()
        size_i = ith_sil.shape[0]
        y_upper = y_lower + size_i
        color = plt.cm.tab10(i / max(1, len(labels_unique)-1))
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil, facecolor=color, edgecolor=color, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_i, str(lab))
        y_lower = y_upper + 10
    ax.axvline(x=sil_avg, color="k", linestyle="--")
    ax.set_title(title + f" (avg={sil_avg:.3f})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Cluster label")
    ax.set_yticks([])
    ax.set_xlim([-0.2, 1.0])
    plt.tight_layout()


def plot_correctness_by_file(ids: List[str], y_bin: List[int], title: str) -> None:
    """Plot a simple chart showing correctness (0/1) for each example/file id."""
    if not ids:
        return
    # Sort by example then filename for readability
    items = sorted(zip(ids, y_bin), key=lambda t: t[0].lower())
    # Keep only 30fps files
    items = [it for it in items if "-30fps" in it[0].lower()]
    labels = [it[0] for it in items]
    vals = [int(it[1]) for it in items]
    y_pos = np.arange(len(labels))
    colors = ['#2ca02c' if v == 1 else '#d62728' for v in vals]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3*len(labels))))
    ax.barh(y_pos, vals, color=colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel('Correctness (0=incorrect, 1=correct)')
    ax.set_title(title)
    # Annotate bars
    for i, v in enumerate(vals):
        ax.text(v + 0.02 if v == 1 else v + 0.02, i + 0.1, str(v), color='black', fontsize=8)
    plt.tight_layout()


def plot_repetition_counts_ax(ax, example: str, ffiles: List[str], seg_csv_path: str, title: str) -> None:
    """Plot per-file repetition counts (correct vs incorrect) for all files in example.

    - Every .npy file in the example is included (even if counts are zero).
    - Uses Segmentation.csv columns: video_id, correctness
    """
    import pandas as pd
    if not os.path.isfile(seg_csv_path):
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv_path}")
    df = pd.read_csv(seg_csv_path, engine="python", sep=None)
    cols = {c.lower(): c for c in df.columns}
    vid_col = cols.get("video_id")
    lab_col = cols.get("correctness")
    if vid_col is None or lab_col is None:
        raise RuntimeError("Segmentation.csv must contain columns 'video_id' and 'correctness'")
    # Normalize video_id to first-6 (lower)
    df["vid6"] = df[vid_col].astype(str).str.strip().str.lower().str[:6]
    df["corr"] = df[lab_col].astype(float).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)

    # Build list of stems for files in this example
    stems = [os.path.splitext(fn)[0] for fn in ffiles]
    # Include both 30fps and 120fps variants
    vid6_list = [s.strip().lower()[:6] for s in stems]

    # Sort stems by numeric ID in pattern PM_### if present
    def sort_key(stem: str) -> int:
        m = re.search(r"pm[_-]?(\d+)", stem.lower())
        return int(m.group(1)) if m else 10**9
    order = sorted(range(len(stems)), key=lambda i: sort_key(stems[i]))
    stems = [stems[i] for i in order]
    vid6_list = [vid6_list[i] for i in order]

    # Count per vid6
    counts = df.groupby("vid6")["corr"].value_counts().unstack(fill_value=0)
    # Ensure columns 0,1 exist
    if 0 not in counts.columns:
        counts[0] = 0
    if 1 not in counts.columns:
        counts[1] = 0
    counts = counts[[0, 1]]

    # Prepare plotting data in file order
    incorrect = []
    correct = []
    labels = []
    for stem, v6 in zip(stems, vid6_list):
        labels.append(f"{example}/{stem}")
        if v6 in counts.index:
            incorrect.append(int(counts.loc[v6, 0]))
            correct.append(int(counts.loc[v6, 1]))
        else:
            incorrect.append(0)
            correct.append(0)

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, incorrect, color="#d62728", alpha=0.8, label="Incorrect")
    ax.barh(y_pos, correct, left=incorrect, color="#2ca02c", alpha=0.8, label="Correct")
    # Shorten labels to PM_###-XXfps and thin out tick labels to reduce clutter
    short_labels = []
    for lab in labels:
        # lab format: Ex*/<stem>; extract example and PM_### id
        try:
            ex_part, stem_part = lab.split('/', 1)
        except ValueError:
            ex_part, stem_part = example, lab
        m = re.search(r"(PM[_-]?\d+)", stem_part, flags=re.IGNORECASE)
        pm_id = m.group(1).upper() if m else stem_part
        short_labels.append(f"{ex_part}/{pm_id}")
    thin = max(1, len(short_labels)//20)
    shown = [lbl if (i % thin == 0) else "" for i, lbl in enumerate(short_labels)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(shown, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Repetition count")
    ax.set_title(title)
    # Show legend only if there is space; caller may hide for other panels
    ax.legend(loc='best', fontsize=8)
    # Annotate incorrect and correct counts per bar (no percentage)
    totals = [i + c for i, c in zip(incorrect, correct)]
    for i, (inc, cor, tot) in enumerate(zip(incorrect, correct, totals)):
        # Incorrect count on left segment (if nonzero)
        if inc > 0:
            ax.text(max(inc * 0.5, 0.05), i, str(inc), va="center", ha="center", fontsize=8, color="#ffffff")
        # Correct count on green segment (if nonzero)
        if cor > 0:
            ax.text(inc + max(cor * 0.5, 0.05), i, str(cor), va="center", ha="center", fontsize=8, color="#ffffff")
    ax.figure.tight_layout()


def _coerce_frame_to_pts3d(frame: np.ndarray) -> np.ndarray:
    """Coerce a single frame to [num_joints, 3] if possible."""
    if frame.ndim == 2:
        if frame.shape[1] >= 3:
            return frame[:, :3]
        if frame.shape[0] >= 3:
            return frame[:3, :].T
        flat = frame.ravel()
    else:
        flat = frame.ravel()
    if flat.size % 3 == 0:
        return flat.reshape(-1, 3)
    if flat.size % 4 == 0:
        return flat.reshape(-1, 4)[:, :3]
    raise ValueError("cannot coerce frame to [N,3]")


def _compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9) -> float:
    """Angle at point b between vectors ba and bc (radians)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0
    v1 /= n1
    v2 /= n2
    cosv = float(np.clip(np.dot(v1, v2), -1.0, 1.0))
    return float(np.arccos(cosv))


def extract_8_angles_sequence(seq: np.ndarray) -> np.ndarray:
    """Return [frames, 8] angle time series from a 3D joint sequence using joints_names.txt indices.

    Angles (vertex is middle joint):
      0) Left shoulder (armpit):   Neck(3) – LeftShoulder(6) – LeftForeArm(8)
      1) Right shoulder (armpit):  Neck(3) – RightShoulder(11) – RightForeArm(13)
      2) Left hip (groin):         Hips(0) – LeftUpLeg(16) – LeftLeg(17)
      3) Right hip (groin):        Hips(0) – RightUpLeg(21) – RightLeg(22)
      4) Left knee:                LeftUpLeg(16) – LeftLeg(17) – LeftFoot(18)
      5) Right knee:               RightUpLeg(21) – RightLeg(22) – RightFoot(23)
      6) Left elbow:               LeftShoulder(6) – LeftForeArm(8) – LeftHand(9)
      7) Right elbow:              RightShoulder(11) – RightForeArm(13) – RightHand(14)
    """
    num_frames = seq.shape[0]
    out = np.zeros((num_frames, 8), dtype=float)
    # Joint triplets (a, b, c) with vertex at b
    triplets = [
        (3, 6, 8),    # L shoulder
        (3, 11, 13),  # R shoulder
        (0, 16, 17),  # L hip
        (0, 21, 22),  # R hip
        (16, 17, 18), # L knee
        (21, 22, 23), # R knee
        (6, 8, 9),    # L elbow
        (11, 13, 14), # R elbow
    ]
    for t in range(num_frames):
        try:
            pts = _coerce_frame_to_pts3d(seq[t])
            if pts.shape[0] < 24:
                if t > 0:
                    out[t] = out[t-1]
                continue
            for k, (ia, ib, ic) in enumerate(triplets):
                out[t, k] = _compute_angle(pts[ia], pts[ib], pts[ic])
        except Exception:
            if t > 0:
                out[t] = out[t-1]
            continue
    return out


def _dtw_distance_multivar(A: np.ndarray, B: np.ndarray) -> float:
    """Multivariate DTW distance with Euclidean local cost.

    A: [Ta, D], B: [Tb, D]
    Returns scalar distance.
    """
    Ta, D = A.shape
    Tb, _ = B.shape
    # Initialize DP with infinities
    dp = np.full((Ta + 1, Tb + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, Ta + 1):
        ai = A[i - 1]
        for j in range(1, Tb + 1):
            bj = B[j - 1]
            cost = float(np.linalg.norm(ai - bj))
            dp[i, j] = cost + min(dp[i - 1, j],    # insertion
                                   dp[i, j - 1],    # deletion
                                   dp[i - 1, j - 1])  # match
    return float(dp[Ta, Tb])


def build_repetition_dataset(
    data_dir: str,
    examples: List[str],
    mode: str,
    seg_csv_path: str,
    use_only_30fps: bool = True,
    max_frames: int = 300,
) -> Tuple[List[np.ndarray], List[int], List[str], List[str]]:
    """Build dataset of segmented repetitions across examples.

    Returns: (series_list, labels_bin, subject_ids, rep_ids)
      - series_list: list of [T, 8] angle series (radians)
      - labels_bin: 0/1 correctness per repetition (>=0.5 -> 1)
      - subject_ids: subject key (e.g., PM_000)
      - rep_ids: human-readable id example/file/repetition_number

    Uses same linking as plot_repetition_counts_ax: map file stems to first-6 video_id,
    and correctness thresholding from Segmentation.csv.
    """
    # Load segmentation CSV similar to plot_repetition_counts_ax
    if not os.path.isfile(seg_csv_path):
        raise FileNotFoundError(f"Segmentation CSV not found: {seg_csv_path}")
    df = pd.read_csv(seg_csv_path, engine="python", sep=None)
    cols = {c.lower(): c for c in df.columns}
    vid_col = cols.get("video_id")
    rep_col = cols.get("repetition_number")
    ff_col = cols.get("first_frame")
    lf_col = cols.get("last_frame")
    lab_col = cols.get("correctness")
    if None in (vid_col, rep_col, ff_col, lf_col, lab_col):
        raise RuntimeError("Segmentation.csv must contain video_id, repetition_number, first_frame, last_frame, correctness")
    df["vid6"] = df[vid_col].astype(str).str.strip().str.lower().str[:6]
    df["corr"] = df[lab_col].astype(float).apply(lambda v: 1 if float(v) >= 0.5 else 0).astype(int)
    df["repno"] = df[rep_col].astype(int)
    df["first"] = df[ff_col].astype(int)
    df["last"] = df[lf_col].astype(int)

    series_list: List[np.ndarray] = []
    labels_bin: List[int] = []
    subject_ids: List[str] = []
    rep_ids: List[str] = []

    for ex in examples:
        folder = os.path.join(data_dir, ("2d_joints" if mode == "2d" else "3d_joints"), ex)
        vframes, ffiles = load_video_frames(folder)
        if not vframes:
            continue
        for i, seq in enumerate(vframes):
            stem = os.path.splitext(ffiles[i])[0] if i < len(ffiles) else str(i)
            if use_only_30fps and ("-30fps" not in stem.lower()):
                continue
            vid6 = _first_six(stem)
            rows = df[df["vid6"] == vid6]
            if rows.empty:
                continue
            # Precompute angle series for full (bounded) sequence
            seq_clip = seq[:max_frames]
            angles = extract_8_angles_sequence(seq_clip)  # [T,8]
            T = angles.shape[0]
            # Emit one sample per repetition row
            for _, r in rows.iterrows():
                start = int(max(0, min(r["first"], T - 1)))
                end = int(max(0, min(r["last"], T - 1)))
                if end <= start:
                    continue
                series = angles[start:end+1]
                series_list.append(series)
                labels_bin.append(int(r["corr"]))
                subj = r[vid_col]
                subject_ids.append(str(subj))
                rep_ids.append(f"{ex}/{stem}#rep{int(r['repno'])}")

    return series_list, labels_bin, subject_ids, rep_ids


def run_1nn_dtw(series_list: List[np.ndarray], labels: List[int]) -> Tuple[float, List[int]]:
    """Leave-one-out 1-NN classification using DTW distance.

    Returns (overall_accuracy, predictions)
    """
    n = len(series_list)
    preds = [0] * n
    correct = 0
    for i in range(n):
        xi = series_list[i]
        best_j = -1
        best_d = np.inf
        for j in range(n):
            if j == i:
                continue
            d = _dtw_distance_multivar(xi, series_list[j])
            if d < best_d:
                best_d = d
                best_j = j
        preds[i] = labels[best_j]
        if preds[i] == labels[i]:
            correct += 1
    return (correct / n) if n > 0 else 0.0, preds


def report_per_subject_accuracy(subject_ids: List[str], labels: List[int], preds: List[int]) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    subs = np.unique(subject_ids)
    for s in subs:
        idxs = [i for i, sid in enumerate(subject_ids) if sid == s]
        if not idxs:
            continue
        c = sum(1 for i in idxs if labels[i] == preds[i])
        acc[str(s)] = c / len(idxs)
    return acc

def main():
    parser = argparse.ArgumentParser(description="1-NN classification and silhouette analysis for exercise correctness")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], default="3d")
    parser.add_argument("--examples", nargs='*', help="Example folders (e.g., Ex1 Ex2 ...). If omitted, autodetect Ex* folders")
    parser.add_argument("--segmentation_csv", type=str, default="Segmentation.csv", help="CSV with ground-truth correctness labels")
    parser.add_argument("--max_frames", type=int, default=300)
    # no classification flags in the original visualization-only flow
    args = parser.parse_args()
    # Build example list
    examples = args.examples if args.examples else []
    if not examples:
        base = os.path.join(args.data_dir, "2d_joints" if args.mode == "2d" else "3d_joints")
        examples = sorted([d for d in os.listdir(base) if d.lower().startswith("ex") and os.path.isdir(os.path.join(base, d))])
        if not examples:
            raise RuntimeError(f"No Ex* folders found in {base}")

    # Original behavior: build visualization grid using Segmentation.csv linkage


if __name__ == "__main__":
    main()


