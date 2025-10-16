import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import json
from pathlib import Path
from typing import (
    Callable,
    List,
    Sequence,
    Tuple,
)

# === Import functions from your DataLoader ===
from DataLoader import (
    load_video_frames,
    get_xycoords_for_plotting,
    get_joint_connections
)

# ============================================================
# 🧭 Pose Utilities (3D)
# Summary: Basic helpers for building stable anatomical frames from 3D keypoints.
# ============================================================
def normalize_vector(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit-norm vector; if near-zero, returns the input vector."""
    n = np.linalg.norm(v)
    return v / n if n > eps else v


def build_torso_frame_3d(
    pelvis: np.ndarray,
    neck: np.ndarray,
    left_hip: np.ndarray,
    right_hip: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute an orthonormal torso frame from 3D joints.

    Returns (origin, right, up, forward), where axes are unit vectors.
    - origin: pelvis
    - up: pelvis→neck direction
    - right: lateral axis (right_hip−left_hip direction, orthonormalized)
    - forward: completes right-handed frame via cross(up, right)
    """
    # Coerce inputs to XYZ vectors (length-3). If 2D, pad Z with 0.
    def _xyz(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v).reshape(-1)
        if v.size >= 3:
            return v[:3].astype(float)
        if v.size == 2:
            return np.array([v[0], v[1], 0.0], dtype=float)
        raise ValueError("joint vector must have at least 2 components")

    pelvis = _xyz(pelvis)
    neck = _xyz(neck)
    left_hip = _xyz(left_hip)
    right_hip = _xyz(right_hip)
    # Base directions
    up = normalize_vector(neck - pelvis)
    lateral = normalize_vector(right_hip - left_hip)
    # Forward via right-handed cross product
    forward = normalize_vector(np.cross(up, lateral))
    # Re-orthonormalize lateral to ensure orthogonality (Gram-Schmidt)
    right = normalize_vector(np.cross(forward, up))
    origin = pelvis
    return origin, right, up, forward

# ============================================================
# 👤 User ROM Normalization & Smoothing
# Summary: Normalize raw angles to [0,1] using per-user ROM and smooth over time.
# ============================================================
class DOFNormalizer:
    """Normalize angles to [0,1] with clamping using per-user ROM bounds."""

    def __init__(self, min_angle_deg: float, max_angle_deg: float, neutral_deg: float = 0.0):
        self.min_angle = float(min_angle_deg)
        self.max_angle = float(max_angle_deg)
        self.neutral = float(neutral_deg)
        self.denom = max(1e-6, self.max_angle - self.min_angle)

    def normalize(self, angle_deg: float) -> float:
        raw = angle_deg - self.neutral
        norm = (raw - (self.min_angle - self.neutral)) / self.denom
        return float(np.clip(norm, 0.0, 1.0))


class DOFSmootherEMA:
    """Exponential moving average smoother for DOF values in [0,1]."""

    def __init__(self, alpha: float = 0.3, initial: float = 0.0):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.state = float(np.clip(initial, 0.0, 1.0))

    def step(self, value: float) -> float:
        value = float(np.clip(value, 0.0, 1.0))
        self.state = self.alpha * value + (1.0 - self.alpha) * self.state
        return self.state

# ============================================================
# 📏 Units & ROM helpers
# ============================================================
def inches_to_meters(x: float) -> float:
    """Convert inches to meters."""
    return float(x) * 0.0254

def meters_to_inches(x: float) -> float:
    """Convert meters to inches."""
    return float(x) / 0.0254


# ============================================================
# 📐 Anthropometric Measurements & Scaling
# Summary: Load and validate comprehensive body measurements for personalized pose transposition.
# ============================================================
class AnthropometricProfile:
    """Comprehensive anthropometric profile for personalized pose scaling."""
    
    def __init__(self, config: dict):
        """Initialize with measurements from config file."""
        anthro = config.get("anthropometrics", {})
        
        # Core measurements
        self.height_in = float(anthro.get("height_in", 70.0))
        self.shoulder_width_in = float(anthro.get("shoulder_width_in", 19.0))
        self.waist_circumference_in = float(anthro.get("waist_circumference_in", 32.0))
        self.chest_circumference_in = float(anthro.get("chest_circumference_in", 38.0))
        self.hip_width_in = float(anthro.get("hip_width_in", 15.0))
        self.head_circumference_in = float(anthro.get("head_circumference_in", 22.0))
        self.torso_length_in = float(anthro.get("torso_length_in", 20.0))
        
        # Bilateral measurements
        self.left_leg_length_in = float(anthro.get("left_leg_length_in", 40.0))
        self.right_leg_length_in = float(anthro.get("right_leg_length_in", 40.0))
        self.left_arm_length_in = float(anthro.get("left_arm_length_in", 24.0))
        self.right_arm_length_in = float(anthro.get("right_arm_length_in", 24.0))
        self.left_hand_length_in = float(anthro.get("left_hand_length_in", 7.0))
        self.right_hand_length_in = float(anthro.get("right_hand_length_in", 7.0))
        self.left_foot_length_in = float(anthro.get("left_foot_length_in", 10.0))
        self.right_foot_length_in = float(anthro.get("right_foot_length_in", 10.0))
        
        # Convert to meters for calculations
        self._convert_to_meters()
        
        # Validate measurements
        self._validate_measurements()
    
    def _convert_to_meters(self):
        """Convert all measurements to meters for internal calculations."""
        self.height_m = inches_to_meters(self.height_in)
        self.shoulder_width_m = inches_to_meters(self.shoulder_width_in)
        self.waist_circumference_m = inches_to_meters(self.waist_circumference_in)
        self.chest_circumference_m = inches_to_meters(self.chest_circumference_in)
        self.hip_width_m = inches_to_meters(self.hip_width_in)
        self.head_circumference_m = inches_to_meters(self.head_circumference_in)
        self.torso_length_m = inches_to_meters(self.torso_length_in)
        
        self.left_leg_length_m = inches_to_meters(self.left_leg_length_in)
        self.right_leg_length_m = inches_to_meters(self.right_leg_length_in)
        self.left_arm_length_m = inches_to_meters(self.left_arm_length_in)
        self.right_arm_length_m = inches_to_meters(self.right_arm_length_in)
        self.left_hand_length_m = inches_to_meters(self.left_hand_length_in)
        self.right_hand_length_m = inches_to_meters(self.right_hand_length_in)
        self.left_foot_length_m = inches_to_meters(self.left_foot_length_in)
        self.right_foot_length_m = inches_to_meters(self.right_foot_length_in)
    
    def _validate_measurements(self):
        """Validate measurement ranges and bilateral differences."""
        # Check for reasonable ranges
        if not (48.0 <= self.height_in <= 84.0):  # 4' to 7' range
            print(f"[warn] Height {self.height_in} inches seems unusual")
        
        if not (12.0 <= self.shoulder_width_in <= 28.0):
            print(f"[warn] Shoulder width {self.shoulder_width_in} inches seems unusual")
        
        # Check bilateral differences (warn if >10% difference)
        leg_diff = abs(self.left_leg_length_in - self.right_leg_length_in) / max(self.left_leg_length_in, self.right_leg_length_in)
        if leg_diff > 0.1:
            print(f"[warn] Significant leg length difference: {leg_diff:.1%}")
        
        arm_diff = abs(self.left_arm_length_in - self.right_arm_length_in) / max(self.left_arm_length_in, self.right_arm_length_in)
        if arm_diff > 0.1:
            print(f"[warn] Significant arm length difference: {arm_diff:.1%}")
    
    def get_limb_scaling_factors(self) -> dict:
        """Return scaling factors for different body parts."""
        # Use average adult proportions as reference
        ref_height = 70.0  # inches
        ref_shoulder_width = 19.0
        ref_arm_length = 24.0
        ref_leg_length = 40.0
        
        return {
            "overall": self.height_in / ref_height,
            "shoulder_width": self.shoulder_width_in / ref_shoulder_width,
            "left_arm": self.left_arm_length_in / ref_arm_length,
            "right_arm": self.right_arm_length_in / ref_arm_length,
            "left_leg": self.left_leg_length_in / ref_leg_length,
            "right_leg": self.right_leg_length_in / ref_leg_length,
            "torso": self.torso_length_in / 20.0,  # reference torso length
        }
    
    def get_limb_lengths_meters(self) -> dict:
        """Return all limb lengths in meters for calculations."""
        return {
            "left_arm": self.left_arm_length_m,
            "right_arm": self.right_arm_length_m,
            "left_leg": self.left_leg_length_m,
            "right_leg": self.right_leg_length_m,
            "left_hand": self.left_hand_length_m,
            "right_hand": self.right_hand_length_m,
            "left_foot": self.left_foot_length_m,
            "right_foot": self.right_foot_length_m,
        }
    
    def print_profile(self):
        """Print comprehensive anthropometric profile."""
        print("\n📏 Anthropometric Profile:")
        print(f"  Height: {self.height_in:.1f}\" ({self.height_m:.2f}m)")
        print(f"  Shoulder Width: {self.shoulder_width_in:.1f}\" ({self.shoulder_width_m:.2f}m)")
        print(f"  Torso Length: {self.torso_length_in:.1f}\" ({self.torso_length_m:.2f}m)")
        print(f"  Left Arm: {self.left_arm_length_in:.1f}\" ({self.left_arm_length_m:.2f}m)")
        print(f"  Right Arm: {self.right_arm_length_in:.1f}\" ({self.right_arm_length_m:.2f}m)")
        print(f"  Left Leg: {self.left_leg_length_in:.1f}\" ({self.left_leg_length_m:.2f}m)")
        print(f"  Right Leg: {self.right_leg_length_in:.1f}\" ({self.right_leg_length_m:.2f}m)")
        
        # Show scaling factors
        factors = self.get_limb_scaling_factors()
        print(f"\n🔧 Scaling Factors:")
        print(f"  Overall: {factors['overall']:.2f}x")
        print(f"  Shoulder: {factors['shoulder_width']:.2f}x")
        print(f"  Left Arm: {factors['left_arm']:.2f}x")
        print(f"  Right Arm: {factors['right_arm']:.2f}x")
        print(f"  Left Leg: {factors['left_leg']:.2f}x")
        print(f"  Right Leg: {factors['right_leg']:.2f}x")


def load_anthropometric_profile(config: dict) -> AnthropometricProfile:
    """Load and validate anthropometric profile from config."""
    return AnthropometricProfile(config)


# ============================================================
# 🔎 Source Sequence Auto-Selection
# Summary: Choose the source subject whose inferred limb proportions
#          best match the user's anthropometrics (arms/legs prioritized).
# ============================================================
def _safe_to_3d(arr: np.ndarray) -> np.ndarray | None:
    try:
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
        if arr.ndim == 2 and arr.shape[0] >= 3:
            return arr[:3, :].T
        flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
        if flat.size % 4 == 0:
            return flat.reshape(-1, 4)[:, :3]
    except Exception:
        pass
    return None


def _safe_to_2d(arr: np.ndarray) -> np.ndarray | None:
    try:
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[:2, :].T
        flat = arr.ravel()
        if flat.size % 2 == 0:
            return flat.reshape(-1, 2)
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)[:, :2]
    except Exception:
        pass
    return None


def _infer_limb_lengths_inches_from_frame(frame: np.ndarray, mode: str, shoulder_width_in_hint: float) -> dict | None:
    """Infer limb lengths from a single frame.

    Returns dict with keys: left_arm_in, right_arm_in, left_leg_in, right_leg_in, shoulder_width_in.
    Uses joint indices per joints_names.txt.
    """
    if mode == "3d":
        pts = _safe_to_3d(frame)
        if pts is None or pts.shape[0] <= 25:
            return None
        # Distances in meters → inches
        def dist(a, b):
            return meters_to_inches(float(np.linalg.norm(pts[a] - pts[b])))
        # Arm: shoulder->elbow + elbow->wrist
        l_arm = dist(6, 8) + dist(8, 9)
        r_arm = dist(11, 13) + dist(13, 14)
        # Leg: hip->knee + knee->ankle (approx using 16->18 via 17; and 21->23 via 22)
        l_leg = dist(16, 17) + dist(17, 18)
        r_leg = dist(21, 22) + dist(22, 23)
        # Shoulder width: LeftArm(7) to RightArm(12)
        sh_w = dist(7, 12)
        return {
            "left_arm_in": l_arm,
            "right_arm_in": r_arm,
            "left_leg_in": l_leg,
            "right_leg_in": r_leg,
            "shoulder_width_in": sh_w,
        }
    else:
        pts = _safe_to_2d(frame)
        if pts is None or pts.shape[0] <= 25:
            return None
        # Estimate pixels-per-inch via shoulder span
        sh_px = float(np.linalg.norm(pts[12] - pts[7])) if pts.shape[0] > 12 else 0.0
        if sh_px <= 1e-6:
            return None
        px_per_in = sh_px / max(1e-6, shoulder_width_in_hint)
        def dist_px_to_in(a, b):
            d_px = float(np.linalg.norm(pts[a] - pts[b]))
            return d_px / px_per_in
        l_arm = dist_px_to_in(6, 8) + dist_px_to_in(8, 9)
        r_arm = dist_px_to_in(11, 13) + dist_px_to_in(13, 14)
        l_leg = dist_px_to_in(16, 17) + dist_px_to_in(17, 18)
        r_leg = dist_px_to_in(21, 22) + dist_px_to_in(22, 23)
        sh_w = sh_px / px_per_in
        return {
            "left_arm_in": l_arm,
            "right_arm_in": r_arm,
            "left_leg_in": l_leg,
            "right_leg_in": r_leg,
            "shoulder_width_in": sh_w,
        }


def _aggregate_inferred_over_sequence(seq: np.ndarray, mode: str, shoulder_width_in_hint: float, max_samples: int = 50) -> dict | None:
    n = seq.shape[0]
    idxs = np.linspace(0, n - 1, num=min(max_samples, n), dtype=int)
    vals = {k: [] for k in ["left_arm_in", "right_arm_in", "left_leg_in", "right_leg_in", "shoulder_width_in"]}
    for t in idxs:
        est = _infer_limb_lengths_inches_from_frame(seq[t], mode, shoulder_width_in_hint)
        if est is None:
            continue
        for k, v in est.items():
            if np.isfinite(v) and v > 0:
                vals[k].append(v)
    if not any(len(v) for v in vals.values()):
        return None
    agg = {k: float(np.median(v)) if len(v) else float("nan") for k, v in vals.items()}
    return agg


def _distance_to_profile(inferred: dict, profile: AnthropometricProfile) -> float:
    # Weights: prioritize arms/legs
    w = {
        "left_arm_in": 3.0,
        "right_arm_in": 3.0,
        "left_leg_in": 3.0,
        "right_leg_in": 3.0,
        "shoulder_width_in": 1.0,
    }
    target = {
        "left_arm_in": profile.left_arm_length_in,
        "right_arm_in": profile.right_arm_length_in,
        "left_leg_in": profile.left_leg_length_in,
        "right_leg_in": profile.right_leg_length_in,
        "shoulder_width_in": profile.shoulder_width_in,
    }
    dist = 0.0
    for k, weight in w.items():
        iv = inferred.get(k, float("nan"))
        tv = target[k]
        if not np.isfinite(iv) or iv <= 0:
            continue
        # Use relative error to be scale-invariant
        rel = abs(iv - tv) / max(1e-6, tv)
        dist += weight * rel
    return float(dist)


def select_best_source_index(
    data_dir: str,
    example: str,
    mode: str,
    profile: AnthropometricProfile,
    max_candidates: int | None = None,
):
    """Return (best_index, filenames, distances) for the example directory.

    Scans all sequences under 2d_joints/ExX or 3d_joints/ExX, infers limb lengths,
    and picks the closest to the user's profile.
    """
    path = os.path.join(
        data_dir,
        "2d_joints" if mode == "2d" else "3d_joints",
        example,
    )
    vframes, ffiles = load_video_frames(path)
    if len(vframes) == 0:
        raise FileNotFoundError(f"No data found for {path}")
    total = len(vframes)
    consider = min(total, max_candidates) if max_candidates else total
    dists = []
    inferred_list = []
    for i in range(consider):
        seq = vframes[i]
        inferred = _aggregate_inferred_over_sequence(seq, mode, profile.shoulder_width_in)
        inferred_list.append(inferred)
        if inferred is None:
            dists.append(float("inf"))
        else:
            d = _distance_to_profile(inferred, profile)
            dists.append(d)
    best_idx = int(np.argmin(dists))
    return best_idx, ffiles, dists, inferred_list


def scale_pose_sequence(pose_seq: np.ndarray, anthropometric_profile: AnthropometricProfile, 
                       mode: str = "3d") -> np.ndarray:
    """Scale pose sequence using anthropometric measurements for personalized transposition."""
    if pose_seq.ndim < 2:
        raise ValueError("pose_seq must be at least [frames, ...]")
    
    # Get scaling factors
    factors = anthropometric_profile.get_limb_scaling_factors()
    limb_lengths = anthropometric_profile.get_limb_lengths_meters()
    
    # Create scaled pose sequence
    scaled_pose = pose_seq.copy()
    
    def _to_pts(arr: np.ndarray) -> np.ndarray:
        """Convert frame to [num_joints, 3] format."""
        if arr.ndim == 2:
            if arr.shape[1] >= 3:
                return arr[:, :3]
            if arr.shape[0] >= 3:
                return arr[:3, :].T
            flat = arr.ravel()
        else:
            flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
        if flat.size % 4 == 0:
            return flat.reshape(-1, 4)[:, :3]
        raise ValueError("cannot coerce frame to [N,3]")
    
    def _to_2d(arr: np.ndarray) -> np.ndarray:
        """Convert frame to [num_joints, 2] format."""
        if arr.ndim == 2:
            if arr.shape[1] >= 2:
                return arr[:, :2]
            if arr.shape[0] >= 2:
                return arr[:2, :].T
            flat = arr.ravel()
        else:
            flat = arr.ravel()
        if flat.size % 2 == 0:
            return flat.reshape(-1, 2)
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)[:, :2]
        raise ValueError("cannot coerce frame to [N,2]")
    
    for t in range(pose_seq.shape[0]):
        try:
            if mode == "3d":
                pts = _to_pts(pose_seq[t])
                if pts.shape[0] < 26:  # Need at least 26 joints
                    continue
                
                # Scale different body parts with appropriate factors
                # Torso joints (0-5): overall scaling
                torso_joints = [0, 1, 2, 3, 4, 5]  # Hips, Spine, Spine1, Neck, Head, Head_end
                for joint_idx in torso_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["overall"]
                
                # Left arm joints (6-10): left arm scaling
                left_arm_joints = [6, 7, 8, 9, 10]  # LeftShoulder, LeftArm, LeftForeArm, LeftHand, LeftHand_end
                for joint_idx in left_arm_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["left_arm"]
                
                # Right arm joints (11-15): right arm scaling
                right_arm_joints = [11, 12, 13, 14, 15]  # RightShoulder, RightArm, RightForeArm, RightHand, RightHand_end
                for joint_idx in right_arm_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["right_arm"]
                
                # Left leg joints (16-20): left leg scaling
                left_leg_joints = [16, 17, 18, 19, 20]  # LeftUpLeg, LeftLeg, LeftFoot, LeftToeBase, LeftToeBase_end
                for joint_idx in left_leg_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["left_leg"]
                
                # Right leg joints (21-25): right leg scaling
                right_leg_joints = [21, 22, 23, 24, 25]  # RightUpLeg, RightLeg, RightFoot, RightToeBase, RightToeBase_end
                for joint_idx in right_leg_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["right_leg"]
                
                # Update the scaled pose
                if scaled_pose[t].ndim == 2 and scaled_pose[t].shape[1] >= 3:
                    scaled_pose[t][:, :3] = pts
                else:
                    # Handle different array shapes
                    flat_pts = pts.ravel()
                    if scaled_pose[t].size >= flat_pts.size:
                        scaled_pose[t].flat[:flat_pts.size] = flat_pts
                        
            else:  # 2D mode
                pts = _to_2d(pose_seq[t])
                if pts.shape[0] < 26:
                    continue
                
                # Apply similar scaling for 2D
                # Torso joints
                torso_joints = [0, 1, 2, 3, 4, 5]
                for joint_idx in torso_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["overall"]
                
                # Left arm joints
                left_arm_joints = [6, 7, 8, 9, 10]
                for joint_idx in left_arm_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["left_arm"]
                
                # Right arm joints
                right_arm_joints = [11, 12, 13, 14, 15]
                for joint_idx in right_arm_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["right_arm"]
                
                # Left leg joints
                left_leg_joints = [16, 17, 18, 19, 20]
                for joint_idx in left_leg_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["left_leg"]
                
                # Right leg joints
                right_leg_joints = [21, 22, 23, 24, 25]
                for joint_idx in right_leg_joints:
                    if joint_idx < pts.shape[0]:
                        pts[joint_idx] *= factors["right_leg"]
                
                # Update the scaled pose
                if scaled_pose[t].ndim == 2 and scaled_pose[t].shape[1] >= 2:
                    scaled_pose[t][:, :2] = pts
                else:
                    flat_pts = pts.ravel()
                    if scaled_pose[t].size >= flat_pts.size:
                        scaled_pose[t].flat[:flat_pts.size] = flat_pts
                        
        except Exception as e:
            # Skip problematic frames
            continue
    
    return scaled_pose


def _get_rom_side(rom_cfg: dict, key: str, side: str, default_min: float = 0.0, default_max: float = 150.0, default_neutral: float = 0.0) -> Tuple[float, float, float]:
    """Return (min,max,neutral) for a given ROM key and side, supporting two schema styles:
    1) Single dict: {min,max,neutral}
    2) Per-side dict: {L:{...}, R:{...}}
    """
    node = rom_cfg.get(key, {})
    if "L" in node or "R" in node:
        node = node.get(side, {})
    mn = float(node.get("min", default_min))
    mx = float(node.get("max", default_max))
    nt = float(node.get("neutral", default_neutral))
    return mn, mx, nt

# ============================================================
# 🦾 Shoulder Angles (3D)
# Summary: Compute shoulder flexion and abduction using torso frame.
# ============================================================
def _project_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Project vector v onto plane with normal n."""
    n = normalize_vector(n)
    return v - np.dot(v, n) * n


def _signed_angle_in_plane(
    v: np.ndarray,
    ref: np.ndarray,
    axis_normal: np.ndarray,
) -> float:
    """Signed angle from ref to v within plane whose normal is axis_normal."""
    v_p = normalize_vector(_project_onto_plane(v, axis_normal))
    r_p = normalize_vector(_project_onto_plane(ref, axis_normal))
    cos_val = float(np.clip(np.dot(v_p, r_p), -1.0, 1.0))
    angle = float(np.arccos(cos_val))
    # Sign by right-hand rule using axis_normal
    sign = np.sign(np.dot(np.cross(r_p, v_p), normalize_vector(axis_normal)))
    return angle * (1.0 if sign >= 0 else -1.0)


def compute_shoulder_angles_3d(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    torso_origin: np.ndarray,
    torso_right: np.ndarray,
    torso_up: np.ndarray,
    torso_forward: np.ndarray,
    in_degrees: bool = False,
) -> Tuple[float, float]:
    """Return (flexion, abduction) angles for a single shoulder in 3D.

    Definitions:
    - Flexion: angle of humerus relative to torso in sagittal plane (up-forward)
      Positive when moving forward/overhead.
    - Abduction: angle in frontal plane (up-right) moving away from torso side.

    Angles are in radians by default; set in_degrees=True to return degrees.
    """
    humerus = shoulder - elbow  # direction from elbow to shoulder
    humerus = -humerus          # shoulder -> elbow vector (humerus axis)
    # Flexion in sagittal plane (up, forward); reference is up
    flexion = _signed_angle_in_plane(humerus, torso_up, torso_right)
    # Abduction in frontal plane (up, right); reference is up
    abduction = _signed_angle_in_plane(humerus, torso_up, torso_forward)
    if in_degrees:
        flexion = float(np.degrees(flexion))
        abduction = float(np.degrees(abduction))
    return flexion, abduction

# ============================================================
# 🤝 Robot Joint Angles (Right Arm, 3D) — degrees
# Summary: Compute [theta0, theta1, theta2, theta3] for right arm.
# - theta0: shoulder horizontal rotation (about torso up)   [-90, +90]
# - theta1: shoulder elevation (0=arm down)                 [0, 120]
# - theta2: elbow flex/extend (0=straight)                  [0, 135]
# - theta3: wrist/end-effector pitch (sagittal)             [-90, +90]
# Notes:
# - Angles are clamped to typical safe mechanical limits.
# - Uses indices: Right shoulder=11, RightForeArm=13 (elbow proxy), RightHand=14 (wrist proxy).
# ============================================================
def compute_right_arm_robot_angles_3d(
    pts: np.ndarray,
    torso_right: np.ndarray,
    torso_up: np.ndarray,
    torso_forward: np.ndarray,
    shoulder_idx: int = 11,
    elbow_idx: int = 13,
    wrist_idx: int = 14,
) -> Tuple[float, float, float, float]:
    # Fetch joints (robust slicing already done upstream for pts)
    if max(shoulder_idx, elbow_idx, wrist_idx) >= pts.shape[0]:
        raise IndexError("Right arm joint indices exceed array shape")

    shoulder = pts[shoulder_idx]
    elbow = pts[elbow_idx]
    wrist = pts[wrist_idx] if wrist_idx < pts.shape[0] else elbow + (elbow - shoulder)

    # Vectors
    humerus = elbow - shoulder  # shoulder -> elbow
    forearm = wrist - elbow     # elbow -> wrist
    if np.linalg.norm(humerus) < 1e-9 or np.linalg.norm(forearm) < 1e-9:
        return 0.0, 0.0, 0.0, 0.0

    h_n = normalize_vector(humerus)
    f_n = normalize_vector(forearm)

    # theta0: horizontal rotation (azimuth around up) — signed angle vs forward in horizontal plane
    h_horiz = h_n - np.dot(h_n, torso_up) * torso_up
    if np.linalg.norm(h_horiz) < 1e-9:
        theta0 = 0.0
    else:
        theta0 = np.degrees(
            _signed_angle_in_plane(h_horiz, torso_forward, torso_up)
        )
    theta0 = float(np.clip(theta0, -90.0, 90.0))

    # theta1: elevation (0=arm down along -up, ~90=horizontal, up to 120)
    # Compute angle between humerus and -up
    cos_elev = float(np.clip(np.dot(h_n, -torso_up), -1.0, 1.0))
    theta1 = float(np.degrees(np.arccos(cos_elev)))
    theta1 = float(np.clip(theta1, 0.0, 120.0))

    # theta2: elbow flexion (0=straight). Angle between -humerus (elbow->shoulder) and forearm (elbow->wrist)
    cos_elbow = float(np.clip(np.dot(normalize_vector(-humerus), f_n), -1.0, 1.0))
    theta2 = float(np.degrees(np.arccos(cos_elbow)))
    theta2 = float(np.clip(theta2, 0.0, 135.0))

    # theta3: wrist/end-effector pitch in sagittal plane (up-forward), signed by right axis
    f_sag = f_n - np.dot(f_n, torso_right) * torso_right
    if np.linalg.norm(f_sag) < 1e-9:
        theta3 = 0.0
    else:
        theta3 = np.degrees(
            _signed_angle_in_plane(f_sag, torso_forward, torso_right)
        )
    theta3 = float(np.clip(theta3, -90.0, 90.0))

    return theta0, theta1, theta2, theta3

def compute_right_arm_robot_angles_3d_sequence(
    pose_seq: np.ndarray,
    shoulder_idx: int = 11,
    elbow_idx: int = 13,
    wrist_idx: int = 14,
) -> np.ndarray:
    """Return [frames, 4] of right-arm robot angles (deg): [theta0, theta1, theta2, theta3]."""
    if pose_seq.ndim < 2:
        raise ValueError("pose_seq must be at least [frames, ...]")

    def _to_pts(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            if arr.shape[1] >= 3:
                return arr[:, :3]
            if arr.shape[0] >= 3:
                return arr[:3, :].T
            flat = arr.ravel()
        else:
            flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
        if flat.size % 4 == 0:
            return flat.reshape(-1, 4)[:, :3]
        raise ValueError("cannot coerce frame to [N,3]")

    num_frames = pose_seq.shape[0]
    out = np.zeros((num_frames, 4), dtype=float)
    for t in range(num_frames):
        try:
            pts = _to_pts(pose_seq[t])
            # Torso frame from pelvis(0), neck(3), hips(16,21)
            if pts.shape[0] <= 21:
                if t > 0:
                    out[t] = out[t-1]
                continue
            origin, right, up, fwd = build_torso_frame_3d(pts[0], pts[3], pts[16], pts[21])
            theta0, theta1, theta2, theta3 = compute_right_arm_robot_angles_3d(
                pts, right, up, fwd, shoulder_idx=shoulder_idx, elbow_idx=elbow_idx, wrist_idx=wrist_idx
            )
            out[t] = [theta0, theta1, theta2, theta3]
        except Exception:
            if t > 0:
                out[t] = out[t-1]
            continue
    return out

# ============================================================
# 🎛️ Shoulder DOF Mapper (3D sequence)
# Summary: Map each 3D frame to L/R flexion & abduction DOFs, normalized & smoothed.
# ============================================================
def map_shoulders_to_dofs_3d_sequence(
    pose_seq: np.ndarray,
    l_sh_idx: int,
    l_el_idx: int,
    r_sh_idx: int,
    r_el_idx: int,
    flex_rom_min_deg: float = 0.0,
    flex_rom_max_deg: float = 150.0,
    abd_rom_min_deg: float = 0.0,
    abd_rom_max_deg: float = 150.0,
    l_flex_neutral_deg: float = 0.0,
    r_flex_neutral_deg: float = 0.0,
    l_abd_neutral_deg: float = 0.0,
    r_abd_neutral_deg: float = 0.0,
    smoothing_alpha: float = 0.3,
) -> np.ndarray:
    """Return [frames, 4] array: [L_flex, R_flex, L_abd, R_abd] in [0,1]."""
    if pose_seq.ndim < 2:
        raise ValueError("pose_seq must be at least [frames, ...]")

    num_frames = pose_seq.shape[0]
    dofs = np.zeros((num_frames, 4), dtype=float)

    # Normalizers & smoothers per DOF
    l_flex_norm = DOFNormalizer(flex_rom_min_deg, flex_rom_max_deg, neutral_deg=l_flex_neutral_deg)
    r_flex_norm = DOFNormalizer(flex_rom_min_deg, flex_rom_max_deg, neutral_deg=r_flex_neutral_deg)
    l_abd_norm = DOFNormalizer(abd_rom_min_deg, abd_rom_max_deg, neutral_deg=l_abd_neutral_deg)
    r_abd_norm = DOFNormalizer(abd_rom_min_deg, abd_rom_max_deg, neutral_deg=r_abd_neutral_deg)
    l_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    r_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    l_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    r_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)

    def _to_pts(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2:
            if arr.shape[1] >= 3:
                return arr[:, :3]
            if arr.shape[0] >= 3:
                return arr[:3, :].T
            flat = arr.ravel()
        else:
            flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
        if flat.size % 4 == 0:
            return flat.reshape(-1, 4)[:, :3]
        raise ValueError("cannot coerce frame to [N,3]")

    for t in range(num_frames):
        try:
            pts = _to_pts(pose_seq[t])
            # Torso anchors (defaults per joints_names.txt): 0 Hips, 3 Neck, 16 L Hip, 21 R Hip
            pelvis = pts[0] if pts.shape[0] > 0 else None
            neck = pts[3] if pts.shape[0] > 3 else None
            l_hip = pts[16] if pts.shape[0] > 16 else None
            r_hip = pts[21] if pts.shape[0] > 21 else None
            if pelvis is None or neck is None or l_hip is None or r_hip is None:
                # carry previous values if torso not available
                if t > 0:
                    dofs[t] = dofs[t - 1]
                continue
            origin, right, up, forward = build_torso_frame_3d(pelvis, neck, l_hip, r_hip)

            # Joints for shoulders and elbows (CLI-provided indices)
            if max(l_sh_idx, l_el_idx, r_sh_idx, r_el_idx) >= pts.shape[0]:
                if t > 0:
                    dofs[t] = dofs[t - 1]
                continue
            l_sh = pts[l_sh_idx]
            l_el = pts[l_el_idx]
            r_sh = pts[r_sh_idx]
            r_el = pts[r_el_idx]

            l_fx_deg, l_ab_deg = compute_shoulder_angles_3d(
                l_sh, l_el, origin, right, up, forward, in_degrees=True
            )
            r_fx_deg, r_ab_deg = compute_shoulder_angles_3d(
                r_sh, r_el, origin, right, up, forward, in_degrees=True
            )

            # Normalize and smooth to [0,1]
            dofs[t, 0] = l_flex_sm.step(l_flex_norm.normalize(l_fx_deg))
            dofs[t, 1] = r_flex_sm.step(r_flex_norm.normalize(r_fx_deg))
            dofs[t, 2] = l_abd_sm.step(l_abd_norm.normalize(l_ab_deg))
            dofs[t, 3] = r_abd_sm.step(r_abd_norm.normalize(r_ab_deg))
        except Exception:
            if t > 0:
                dofs[t] = dofs[t - 1]
            continue

    return dofs

# ============================================================
# 🎛️ Shoulder DOF Mapper (2D sequence)
# Summary: Map each 2D frame to L/R flexion & abduction DOFs using shoulder-width normalization.
# ============================================================
def map_shoulders_to_dofs_2d_sequence(
    pose_seq: np.ndarray,
    l_sh_idx: int,
    l_el_idx: int,
    r_sh_idx: int,
    r_el_idx: int,
    shoulder_width_in: float = 15.0,
    pixels_per_meter_hint: float | None = None,
    flex_rom_min_deg: float = 0.0,
    flex_rom_max_deg: float = 150.0,
    abd_rom_min_deg: float = 0.0,
    abd_rom_max_deg: float = 150.0,
    l_flex_neutral_deg: float = 0.0,
    r_flex_neutral_deg: float = 0.0,
    l_abd_neutral_deg: float = 0.0,
    r_abd_neutral_deg: float = 0.0,
    smoothing_alpha: float = 0.3,
) -> np.ndarray:
    """Return [frames, 4] array: [L_flex, R_flex, L_abd, R_abd] in [0,1] from 2D joints.

    Heuristics:
    - Flexion proxy: vertical displacement of elbow above shoulder (pixels) normalized by shoulder width.
    - Abduction proxy: lateral displacement of elbow from the shoulder line normalized by shoulder width.
    """
    if pose_seq.ndim < 2:
        raise ValueError("pose_seq must be at least [frames, ...]")

    num_frames = pose_seq.shape[0]
    dofs = np.zeros((num_frames, 4), dtype=float)

    l_flex_norm = DOFNormalizer(flex_rom_min_deg, flex_rom_max_deg, neutral_deg=l_flex_neutral_deg)
    r_flex_norm = DOFNormalizer(flex_rom_min_deg, flex_rom_max_deg, neutral_deg=r_flex_neutral_deg)
    l_abd_norm = DOFNormalizer(abd_rom_min_deg, abd_rom_max_deg, neutral_deg=l_abd_neutral_deg)
    r_abd_norm = DOFNormalizer(abd_rom_min_deg, abd_rom_max_deg, neutral_deg=r_abd_neutral_deg)
    l_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    r_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    l_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)
    r_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)

    def _to_2d(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[:2, :].T
        flat = arr.ravel()
        if flat.size % 2 == 0:
            return flat.reshape(-1, 2)
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)[:, :2]
        raise ValueError("cannot coerce frame to [N,2]")

    shoulder_width_m = inches_to_meters(shoulder_width_in)
    for t in range(num_frames):
        try:       
            pts2 = _to_2d(pose_seq[t])
            if max(l_sh_idx, l_el_idx, r_sh_idx, r_el_idx) >= pts2.shape[0]:
                if t > 0:
                    dofs[t] = dofs[t - 1]
                continue
            # Pixel shoulder span estimate per frame (LeftArm 7 to RightArm 12)
            if pts2.shape[0] > max(7, 12):
                sh_w_px = float(np.linalg.norm(pts2[12] - pts2[7]))
            else:
                sh_w_px = 1.0
            if sh_w_px <= 1e-6:
                if t > 0:
                    dofs[t] = dofs[t - 1]
                continue

            # Vertical up is decreasing y in most image coords; use absolute displacement magnitude
            l_sh, l_el = pts2[l_sh_idx], pts2[l_el_idx]
            r_sh, r_el = pts2[r_sh_idx], pts2[r_el_idx]
            l_flex_proxy_deg = float(np.degrees(np.arctan2(abs(l_sh[1] - l_el[1]), sh_w_px))) * 180.0 / 90.0
            r_flex_proxy_deg = float(np.degrees(np.arctan2(abs(r_sh[1] - r_el[1]), sh_w_px))) * 180.0 / 90.0
            l_abd_proxy_deg = float(np.degrees(np.arctan2(abs(l_sh[0] - l_el[0]), sh_w_px))) * 180.0 / 90.0
            r_abd_proxy_deg = float(np.degrees(np.arctan2(abs(r_sh[0] - r_el[0]), sh_w_px))) * 180.0 / 90.0

            dofs[t, 0] = l_flex_sm.step(l_flex_norm.normalize(l_flex_proxy_deg))
            dofs[t, 1] = r_flex_sm.step(r_flex_norm.normalize(r_flex_proxy_deg))
            dofs[t, 2] = l_abd_sm.step(l_abd_norm.normalize(l_abd_proxy_deg))
            dofs[t, 3] = r_abd_sm.step(r_abd_norm.normalize(r_abd_proxy_deg))
        except Exception:
            if t > 0:
                dofs[t] = dofs[t - 1]
            continue

    return dofs
# ============================================================
# 🧠 PID Controller
# Summary: Provides a vectorized PID controller to compute control
#          signals for multiple DOFs simultaneously.
# ============================================================
class PIDController:
    """Simple vectorized PID controller for multiple DOFs."""

    def __init__(self, kp: float, ki: float, kd: float, dof: int, timestep: float = 0.01) -> None:
        """Initialize per-DOF PID gains and state buffers."""
        # Validate configuration and initialize per-DOF gains/state
        if dof <= 0:
            raise ValueError("dof must be positive")
        self.kp = np.full(dof, kp, dtype=float)
        self.ki = np.full(dof, ki, dtype=float)
        self.kd = np.full(dof, kd, dtype=float)
        self.timestep = float(timestep)
        self.integral = np.zeros(dof, dtype=float)
        self.prev_error = np.zeros(dof, dtype=float)

    def compute(self, target: np.ndarray, feedback: np.ndarray) -> np.ndarray:
        """Compute control output given target and feedback vectors."""
        # Standard PID: u = Kp*e + Ki*∫e dt + Kd*de/dt
        if target.shape != feedback.shape:
            raise ValueError("target and feedback must have the same shape")
        error = target - feedback
        self.integral += error * self.timestep
        derivative = (error - self.prev_error) / self.timestep
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


# ============================================================
# 🤖 Soft Physiotherapy Robot
# Summary: Simulates a simple soft robot with per-DOF state updated
#          by control outputs and constrained by actuator limits.
# ============================================================
class SoftPhysioRobot:
    """Minimal soft robot model driven by a provided control function."""

    def __init__(
        self,
        dof: int,
        control_function: Callable[[np.ndarray, np.ndarray, int], np.ndarray],
        actuator_limits=None,
        timestep: float = 0.01,
    ):
        """Configure robot DOF, control callback, timestep, and actuator limits."""
        # Constrain DOF to a sane range and initialize state/limits
        if dof <= 0:
            raise ValueError("dof must be positive")
        self.dof = int(np.clip(dof, 1, 10))
        self.control_function = control_function
        self.timestep = float(timestep)
        self.state = np.zeros(self.dof, dtype=float)
        self.feedback = np.zeros(self.dof, dtype=float)
        self.actuator_limits = (
            actuator_limits
            if actuator_limits is not None
            else np.array([[0.0, 1.0]] * self.dof, dtype=float)
        )
        if self.actuator_limits.shape != (self.dof, 2):
            raise ValueError("actuator_limits must have shape (dof, 2)")

    def actuate(self, input_vector: np.ndarray):
        """Apply one control step and update internal state; returns (cmds, state)."""
        # Compute control command then clip to per-DOF actuator limits
        if input_vector.shape[0] != self.dof:
            raise ValueError("input_vector length must match dof")
        actuator_commands = self.control_function(input_vector, self.feedback, self.dof)
        actuator_commands = np.asarray(actuator_commands, dtype=float)
        if actuator_commands.shape[0] != self.dof:
            raise ValueError("control_function must return vector of length dof")
        actuator_min = self.actuator_limits[:, 0]
        actuator_max = self.actuator_limits[:, 1]
        actuator_commands = np.clip(actuator_commands, actuator_min, actuator_max)
        # First-order discrete integration to update internal state
        self.state = self.state + actuator_commands * self.timestep
        self.feedback = self.state.copy()
        return actuator_commands, self.state

    def perform_exercise(self, input_sequence, exercise_name: str = "default", real_time: bool = False):
        """Run a sequence of inputs through the controller; optionally in real-time."""
        # Replay a sequence of target inputs; optionally sleep to emulate real-time
        print(f"\n🏋️ Starting physiotherapy exercise: {exercise_name}")
        history = []
        for t, inputs in enumerate(input_sequence):
            actuator_cmds, next_state = self.actuate(np.array(inputs, dtype=float))
            history.append((t, np.asarray(inputs, dtype=float), actuator_cmds, next_state))
            if real_time:
                time.sleep(self.timestep)
        print(f"✅ Completed exercise: {exercise_name}\n")
        return history


# ============================================================
# ⚙️ Control Strategies
# Summary: Implements basic control policies (P and PID factory) used
#          to generate actuator commands for the robot.
# ============================================================
def proportional_control(inputs: np.ndarray, feedback: np.ndarray, dof: int) -> np.ndarray:
    """Return proportional error scaled by a fixed gain for the first DOF entries."""
    # Simple P-controller used as a baseline
    gain = 0.6
    return gain * (inputs[:dof] - feedback[:dof])

def make_pid_control(
    kp: float = 0.8,
    ki: float = 0.1,
    kd: float = 0.05,
    dof: int = 6,
    timestep: float = 0.05,
):
    """Factory that returns a PID-based control function bound to given gains."""
    # Factory to create a closure around a PIDController instance
    pid = PIDController(kp, ki, kd, dof, timestep)

    def pid_control(inputs: np.ndarray, feedback: np.ndarray, dof: int) -> np.ndarray:
        # Compute PID control per call using current feedback
        return pid.compute(inputs[:dof], feedback[:dof])

    return pid_control


# ============================================================
# 🏋️ Pose Loader + Mapping to Robot Inputs
# Summary: Loads pose sequences from disk and reduces high-dimensional
#          pose features into DOF-sized robot inputs with normalization.
# ============================================================
def load_pose_sequence(
    data_dir: str,
    idx: int = 0,
    example: str = "Ex1",
    mode: str = "2d",
    max_frames: int = 800,
):
    """Load one pose sequence by index from the specified example and mode."""
    # Select 2D or 3D pose directory and return a bounded-length sequence
    if mode not in {"2d", "3d"}:
        raise ValueError("mode must be '2d' or '3d'")
    path = os.path.join(
        data_dir,
        "2d_joints" if mode == "2d" else "3d_joints",
        example,
    )
    vframes, _ = load_video_frames(path)
    if len(vframes) == 0:
        raise FileNotFoundError(f"No data found for {path}")
    if not (0 <= idx < len(vframes)):
        raise IndexError(
            f"idx {idx} out of range for available sequences: {len(vframes)}"
        )
    return vframes[idx][:max_frames]


def convert_pose_to_robot_inputs(
    pose_sequence: np.ndarray,
    dof: int,
) -> np.ndarray:
    """Aggregate pose features into DOF-sized inputs and normalize each DOF over time."""
    # Reduce high-dimensional pose features into DOF groups and normalize per DOF over time
    frames, joints, dims = pose_sequence.shape
    print('Frames:', frames, ' Joints:', joints, ' Dims:', dims)
    flat_pose = pose_sequence.reshape(frames, joints * dims)
    if dof <= 0:
        raise ValueError("dof must be positive")
    feature_indices = np.array_split(np.arange(flat_pose.shape[1]), dof)
    grouped_means = [
        np.mean(flat_pose[:, idxs], axis=1) if len(idxs) > 0 else np.zeros(frames)
        for idxs in feature_indices
    ]
    robot_inputs = np.stack(grouped_means, axis=1).astype(float)
    col_min = robot_inputs.min(axis=0)
    col_max = robot_inputs.max(axis=0)
    denom = (col_max - col_min) + 1e-8
    robot_inputs = (robot_inputs - col_min) / denom
    return robot_inputs


# ============================================================
# 🎥 Visualization: Pose + Robot State (Side by Side)
# Summary: Renders the human pose and the robot's per-DOF states side
#          by side over time for quick qualitative assessment.
# ============================================================
def visualize_pose_and_robot(
    pose_seq: np.ndarray,
    robot_history: Sequence[Tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    mode: str = "2d",
    interval: int = 50,
    anthropometric_profile: AnthropometricProfile = None,
):
    """Animate pose and robot DOF states in a two-panel matplotlib figure."""
    # Side-by-side plot: human pose on left, robot actuator states on right
    connections = get_joint_connections()
    num_frames = min(len(robot_history), pose_seq.shape[0])
    if num_frames == 0:
        return
    dof = robot_history[0][1].shape[0]

    # Create axes; use 3D projection for pose when requested
    if mode == "3d":
        fig = plt.figure(figsize=(12, 6))
        ax_pose = fig.add_subplot(1, 2, 1, projection='3d')
        ax_robot = fig.add_subplot(1, 2, 2)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax_pose, ax_robot = axes
    
    # Add anthropometric scaling info if available
    if anthropometric_profile is not None:
        factors = anthropometric_profile.get_limb_scaling_factors()
        scaling_info = f"Scaling: H:{factors['overall']:.2f}x, L-Arm:{factors['left_arm']:.2f}x, R-Arm:{factors['right_arm']:.2f}x"
        fig.suptitle(f"Physiotherapy Robot Simulation - {scaling_info}", fontsize=10)
    
    plt.ion()

    for t in range(0, num_frames, 2):
        ax_pose.cla()
        ax_robot.cla()

        if mode == "2d":
            joints = pose_seq[t]
            x, y = joints[:, 0], joints[:, 1]
            xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
            ax_pose.scatter(x, y, c='red', s=30)
            for s, e in connections:
                if s < len(joints) and e < len(joints):
                    ax_pose.plot([x[s], x[e]], [y[s], y[e]], 'b-')
            ax_pose.set_title(f"Human Pose Frame {t}")
            ax_pose.set_xlim([xmin - 100, xmax + 100])
            ax_pose.set_ylim([ymax + 100, ymin - 100])
        else:
            joints = pose_seq[t]
            # Normalize shape to [num_joints, 3] robustly
            try:
                if joints.ndim == 2:
                    # Common cases: [..., 3] or [3, ...] or [..., 4] (xyz + conf)
                    if joints.shape[1] >= 3:
                        pts = joints[:, :3]
                    elif joints.shape[0] >= 3:
                        pts = joints[:3, :].T
                    else:
                        # As a last resort, attempt flat reshape if possible
                        flat = joints.ravel()
                        if flat.size % 3 == 0:
                            pts = flat.reshape(-1, 3)
                        elif flat.size % 4 == 0:
                            pts = flat.reshape(-1, 4)[:, :3]
                        else:
                            # Cannot infer 3D layout; skip this frame
                            continue
                elif joints.ndim == 1:
                    flat = joints
                    if flat.size % 3 == 0:
                        pts = flat.reshape(-1, 3)
                    elif flat.size % 4 == 0:
                        pts = flat.reshape(-1, 4)[:, :3]
                    else:
                        continue
                else:
                    continue
            except Exception:
                # Skip on any unexpected shape/reshape error
                continue

            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            xmin, xmax = np.min(x), np.max(x)
            ymin, ymax = np.min(y), np.max(y)
            zmin, zmax = np.min(z), np.max(z)
            ax_pose.scatter(x, y, z, c='red', s=25, alpha=0.9)
            for s, e in connections:
                if s < len(pts) and e < len(pts):
                    ax_pose.plot([x[s], x[e]], [y[s], y[e]], [z[s], z[e]], 'b-', lw=1.5)
            ax_pose.set_title(f"Human Pose Frame {t} (3D)")
            # Adaptive padding based on data spread
            rng = max(xmax - xmin, ymax - ymin, zmax - zmin)
            pad = max(1e-3, 0.1 * rng)
            ax_pose.set_xlim([xmin - pad, xmax + pad])
            ax_pose.set_ylim([ymin - pad, ymax + pad])
            ax_pose.set_zlim([zmin - pad, zmax + pad])
            # Helpful default view
            ax_pose.view_init(elev=20, azim=-60)

        _, _, act_cmds, next_state = robot_history[t]
        ax_robot.bar(np.arange(dof), next_state, color='orange', alpha=0.7)
        ax_robot.set_ylim(0, 2)
        ax_robot.set_title("Robot Actuator States")
        ax_robot.set_xlabel("DOF Index")
        ax_robot.set_ylabel("State (normalized)")

        plt.pause(interval / 1000)

    plt.ioff()
    plt.show()

# ============================================================
# 🧩 Main Integrated Entry Point
# Summary: Parses CLI options, runs the simulation using the chosen
#          control strategy, and visualizes results.
# ============================================================
def main():
    """Parse CLI arguments, run the simulation, and visualize the results."""
    # CLI to configure data source, control strategy, and simulation options
    parser = argparse.ArgumentParser(
        description="Physiotherapy Robot Simulation using Pose Data",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Base directory containing pose data",
    )
    parser.add_argument(
        "--example",
        type=str,
        default="Ex1",
        help="Exercise example folder (Ex1/Ex2/...)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["2d", "3d"],
        default="2d",
        help="Pose data dimensionality",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=5,
        help="Sequence index to load",
    )
    parser.add_argument(
        "--dof",
        type=int,
        default=8,
        help="Robot degrees of freedom (1-10)",
    )
    parser.add_argument(
        "--control",
        type=str,
        choices=["pid", "proportional"],
        default="pid",
        help="Control strategy",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.05,
        help="Simulation timestep in seconds",
    )
    parser.add_argument(
        "--interval_ms",
        type=int,
        default=100,
        help="Visualization frame interval in ms",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config for joints/ROM/smoothing (default: physio_config.json)",
    )
    parser.add_argument(
        "--debug_angles",
        action="store_true",
        help="Print sample 3D shoulder flexion/abduction angles",
    )
    parser.add_argument(
        "--demo_user_dof",
        action="store_true",
        help="Demo per-user DOF normalization and smoothing for shoulders",
    )
    parser.add_argument(
        "--export_robot_angles",
        action="store_true",
        help="Print sample right-arm robot joint angles (deg) on first 3D frame",
    )
    parser.add_argument(
        "--use_shoulder_mapper",
        action="store_true",
        help="Use 3D shoulder mapper as robot inputs (overrides default mapping)",
    )
    parser.add_argument(
        "--l_sh_idx",
        type=int,
        default=6,
        help="Index of LeftShoulder joint in pose array",
    )
    parser.add_argument(
        "--l_el_idx",
        type=int,
        default=8,
        help="Index of Left elbow proxy (e.g., LeftForeArm) in pose array",
    )
    parser.add_argument(
        "--r_sh_idx",
        type=int,
        default=11,
        help="Index of RightShoulder joint in pose array",
    )
    parser.add_argument(
        "--r_el_idx",
        type=int,
        default=13,
        help="Index of Right elbow proxy (e.g., RightForeArm) in pose array",
    )
    parser.add_argument(
        "--use_anthropometric_scaling",
        action="store_true",
        default=True,
        help="Apply anthropometric scaling for personalized pose transposition",
    )
    parser.add_argument(
        "--auto_select_source",
        action="store_true",
        help="Auto-select the best source sequence by matching anthropometrics",
    )
    parser.add_argument(
        "--max_source_candidates",
        type=int,
        default=None,
        help="Limit number of candidate sequences to consider during auto-selection",
    )
    args = parser.parse_args()

    print("\n=== Physiotherapy Robot Simulation using Real Pose Data ===")

    # Load configuration (optional) and override relevant args
    def load_config(path: str | None) -> dict:
        candidates = []
        if path:
            candidates.append(Path(path))
        candidates.append(Path("physio_config.json"))
        for p in candidates:
            if p.is_file():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        cfg = json.load(f)
                        print(f"Using config: {p}")
                        return cfg
                except Exception as e:
                    print(f"[warn] failed to read config {p}: {e}")
        return {}

    cfg = load_config(args.config)
    # Override joint indices if present
    joints_cfg = cfg.get("joints", {})
    args.l_sh_idx = int(joints_cfg.get("l_sh_idx", args.l_sh_idx))
    args.l_el_idx = int(joints_cfg.get("l_el_idx", args.l_el_idx))
    args.r_sh_idx = int(joints_cfg.get("r_sh_idx", args.r_sh_idx))
    args.r_el_idx = int(joints_cfg.get("r_el_idx", args.r_el_idx))
    # ROM and smoothing for demo path (supports per-side or shared configs)
    rom_cfg = cfg.get("rom_deg", {})
    # Load comprehensive anthropometric profile
    anthropometric_profile = load_anthropometric_profile(cfg)
    shoulder_width_in = anthropometric_profile.shoulder_width_in
    shoulder_width_m = anthropometric_profile.shoulder_width_m
    smoothing_alpha = float(cfg.get("smoothing", {}).get("alpha", 0.3))
    
    # Print anthropometric profile for verification
    if debug_print_torso:
        anthropometric_profile.print_profile()

    # Per-side ROM and neutrals
    l_fx_min, l_fx_max, l_fx_neu = _get_rom_side(rom_cfg, "shoulder_flexion", "L")
    r_fx_min, r_fx_max, r_fx_neu = _get_rom_side(rom_cfg, "shoulder_flexion", "R")
    l_ab_min, l_ab_max, l_ab_neu = _get_rom_side(rom_cfg, "shoulder_abduction", "L")
    r_ab_min, r_ab_max, r_ab_neu = _get_rom_side(rom_cfg, "shoulder_abduction", "R")
    # Input mapping behavior (default: always use shoulder mapper in 3D)
    use_shoulder_mapper = bool(cfg.get("inputs", {}).get("use_shoulder_mapper", True))
    # Debug/prints behavior (default ON while coding)
    dbg_cfg = cfg.get("debug", {})
    debug_print_torso = bool(dbg_cfg.get("print_torso", True))
    debug_print_angles = bool(dbg_cfg.get("print_angles", True)) or args.debug_angles
    debug_demo_user_dof = bool(dbg_cfg.get("print_dof_demo", True)) or args.demo_user_dof

    # If requested, auto-select the best matching source sequence (per exercise)
    if args.auto_select_source:
        try:
            best_idx, ffiles, dists, inferred_list = select_best_source_index(
                args.data_dir, args.example, args.mode, anthropometric_profile, max_candidates=args.max_source_candidates
            )
            print("\n🔎 Auto-selected source sequence:")
            print(f"  Example: {args.example}")
            print(f"  Candidates: {len(ffiles)} | Considered: {len(dists)}")
            print(f"  Selected index: {best_idx} | File: {ffiles[best_idx] if best_idx < len(ffiles) else 'N/A'}")
            print(f"  Distance: {dists[best_idx]:.3f}")
            inf = inferred_list[best_idx]
            if inf is not None:
                print("  Inferred (in):", {k: round(v, 2) for k, v in inf.items() if np.isfinite(v)})
            args.idx = best_idx
        except Exception as e:
            print(f"[warn] Auto-selection failed: {e}. Falling back to provided idx={args.idx}")

    pose_seq = load_pose_sequence(args.data_dir, args.idx, args.example, args.mode)
    
    # Apply anthropometric scaling for personalized transposition
    if args.use_anthropometric_scaling and args.mode in ["2d", "3d"]:
        print(f"🔧 Applying anthropometric scaling for personalized transposition...")
        pose_seq = scale_pose_sequence(pose_seq, anthropometric_profile, args.mode)
        print(f"✅ Pose sequence scaled using personal measurements")
    elif not args.use_anthropometric_scaling:
        print(f"⚠️  Anthropometric scaling disabled - using original pose data")

    # Optional: Build torso frame in 3D mode for downstream shoulder DOF mapping
    if args.mode == "3d":
        try:
            # Expect pose_seq: [frames, joints, 3]
            sample = pose_seq[0]
            pelvis = sample[0]            # Hips
            neck = sample[3]              # Neck
            left_hip = sample[16]         # LeftUpLeg root
            right_hip = sample[21]        # RightUpLeg root
            origin, right, up, forward = build_torso_frame_3d(
                pelvis, neck, left_hip, right_hip
            )
            # Minimal diagnostic output to verify
            if debug_print_torso:
                print("Torso frame:",
                      "origin=", origin,
                      "| right=", np.round(right, 3),
                      "| up=", np.round(up, 3),
                      "| fwd=", np.round(forward, 3))

            if debug_print_angles:
                # Coerce sample to [num_joints, 3] for robust indexing
                def _to_pts(arr: np.ndarray) -> np.ndarray:
                    if arr.ndim == 2:
                        if arr.shape[1] >= 3:
                            return arr[:, :3]
                        if arr.shape[0] >= 3:
                            return arr[:3, :].T
                        flat = arr.ravel()
                    else:
                        flat = arr.ravel()
                    if flat.size % 3 == 0:
                        return flat.reshape(-1, 3)
                    if flat.size % 4 == 0:
                        return flat.reshape(-1, 4)[:, :3]
                    raise ValueError("cannot coerce sample to [N,3]")

                pts = _to_pts(sample)
                # Joint indices (CLI-overridable)
                idx_l_sh, idx_l_el = args.l_sh_idx, args.l_el_idx
                idx_r_sh, idx_r_el = args.r_sh_idx, args.r_el_idx
                if max(idx_l_sh, idx_l_el, idx_r_sh, idx_r_el) < pts.shape[0]:
                    l_sh = pts[idx_l_sh]
                    l_el = pts[idx_l_el]
                    r_sh = pts[idx_r_sh]
                    r_el = pts[idx_r_el]
                    l_flex, l_abd = compute_shoulder_angles_3d(
                        l_sh, l_el, origin, right, up, forward, in_degrees=True
                    )
                    r_flex, r_abd = compute_shoulder_angles_3d(
                        r_sh, r_el, origin, right, up, forward, in_degrees=True
                    )
                    print(
                        "Shoulder angles (deg):",
                        "L flex=", round(l_flex, 1), "L abd=", round(l_abd, 1),
                        "| R flex=", round(r_flex, 1), "R abd=", round(r_abd, 1),
                    )
                else:
                    print("[warn] debug_angles skipped: joint indices exceed array shape", pts.shape)

            if args.export_robot_angles:
                # Compute right-arm robot angles (deg) from first frame
                try:
                    pts = _to_pts(sample)
                    if pts.shape[0] > max(21, 14):
                        _, right, up, fwd = build_torso_frame_3d(pts[0], pts[3], pts[16], pts[21])
                        th0, th1, th2, th3 = compute_right_arm_robot_angles_3d(
                            pts, right, up, fwd, shoulder_idx=args.r_sh_idx, elbow_idx=args.r_el_idx, wrist_idx=14
                        )
                        print(
                            "Right-arm robot angles (deg):",
                            f"theta0={th0:.1f}, theta1={th1:.1f}, theta2={th2:.1f}, theta3={th3:.1f}"
                        )
                    else:
                        print("[warn] export_robot_angles skipped: insufficient joints")
                except Exception as e:
                    print("[warn] export_robot_angles failed:", e)

            if debug_demo_user_dof:
                # Simple demonstration of per-user normalization and smoothing
                # Therapist-provided ROM and smoothing from config (per-side aware)
                l_flex_norm = DOFNormalizer(min_angle_deg=l_fx_min, max_angle_deg=l_fx_max, neutral_deg=l_fx_neu)
                r_flex_norm = DOFNormalizer(min_angle_deg=r_fx_min, max_angle_deg=r_fx_max, neutral_deg=r_fx_neu)
                l_abd_norm = DOFNormalizer(min_angle_deg=l_ab_min, max_angle_deg=l_ab_max, neutral_deg=l_ab_neu)
                r_abd_norm = DOFNormalizer(min_angle_deg=r_ab_min, max_angle_deg=r_ab_max, neutral_deg=r_ab_neu)
                l_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
                l_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)
                r_flex_sm = DOFSmootherEMA(alpha=smoothing_alpha)
                r_abd_sm = DOFSmootherEMA(alpha=smoothing_alpha)

                # Use the same first frame sample for a quick demo
                pts = _to_pts(sample)
                if max(args.l_sh_idx, args.l_el_idx, args.r_sh_idx, args.r_el_idx) < pts.shape[0]:
                    l_sh, l_el = pts[args.l_sh_idx], pts[args.l_el_idx]
                    r_sh, r_el = pts[args.r_sh_idx], pts[args.r_el_idx]
                    l_fx, l_ab = compute_shoulder_angles_3d(l_sh, l_el, origin, right, up, forward, in_degrees=True)
                    r_fx, r_ab = compute_shoulder_angles_3d(r_sh, r_el, origin, right, up, forward, in_degrees=True)
                    # Normalize and smooth
                    l_fx_n = l_flex_sm.step(l_flex_norm.normalize(l_fx))
                    l_ab_n = l_abd_sm.step(l_abd_norm.normalize(l_ab))
                    r_fx_n = r_flex_sm.step(r_flex_norm.normalize(r_fx))
                    r_ab_n = r_abd_sm.step(r_abd_norm.normalize(r_ab))
                    print(
                        "DOF demo (normalized & smoothed):",
                        "L flex=", round(l_fx_n, 3), "L abd=", round(l_ab_n, 3),
                        "| R flex=", round(r_fx_n, 3), "R abd=", round(r_ab_n, 3),
                    )
                else:
                    print("[warn] demo_user_dof skipped: joint indices exceed array shape", pts.shape)
        except Exception as e:
            print("[warn] torso frame build skipped:", e)
            
    robot_inputs = convert_pose_to_robot_inputs(pose_seq, args.dof)
    if args.mode == "3d" and use_shoulder_mapper:
        try:
            dof_seq = map_shoulders_to_dofs_3d_sequence(
                pose_seq,
                l_sh_idx=args.l_sh_idx,
                l_el_idx=args.l_el_idx,
                r_sh_idx=args.r_sh_idx,
                r_el_idx=args.r_el_idx,
                flex_rom_min_deg=min(l_fx_min, r_fx_min),
                flex_rom_max_deg=max(l_fx_max, r_fx_max),
                abd_rom_min_deg=min(l_ab_min, r_ab_min),
                abd_rom_max_deg=max(l_ab_max, r_ab_max),
                l_flex_neutral_deg=l_fx_neu,
                r_flex_neutral_deg=r_fx_neu,
                l_abd_neutral_deg=l_ab_neu,
                r_abd_neutral_deg=r_ab_neu,
                smoothing_alpha=smoothing_alpha,
            )
            if args.dof <= 4:
                robot_inputs = dof_seq[:, :args.dof]
            else:
                pad = np.zeros((dof_seq.shape[0], args.dof - 4), dtype=float)
                robot_inputs = np.concatenate([dof_seq, pad], axis=1)
            print("Using 3D shoulder mapper for robot inputs.")
        except Exception as e:
            print("[warn] shoulder mapper failed, using default inputs:", e)
    elif args.mode == "2d" and use_shoulder_mapper:
        try:
            dof_seq = map_shoulders_to_dofs_2d_sequence(
                pose_seq,
                l_sh_idx=args.l_sh_idx,
                l_el_idx=args.l_el_idx,
                r_sh_idx=args.r_sh_idx,
                r_el_idx=args.r_el_idx,
                shoulder_width_in=shoulder_width_in,
                flex_rom_min_deg=min(l_fx_min, r_fx_min),
                flex_rom_max_deg=max(l_fx_max, r_fx_max),
                abd_rom_min_deg=min(l_ab_min, r_ab_min),
                abd_rom_max_deg=max(l_ab_max, r_ab_max),
                l_flex_neutral_deg=l_fx_neu,
                r_flex_neutral_deg=r_fx_neu,
                l_abd_neutral_deg=l_ab_neu,
                r_abd_neutral_deg=r_ab_neu,
                smoothing_alpha=smoothing_alpha,
            )
            if args.dof <= 4:
                robot_inputs = dof_seq[:, :args.dof]
            else:
                pad = np.zeros((dof_seq.shape[0], args.dof - 4), dtype=float)
                robot_inputs = np.concatenate([dof_seq, pad], axis=1)
            print("Using 2D shoulder mapper for robot inputs.")
        except Exception as e:
            print("[warn] 2D shoulder mapper failed, using default inputs:", e)

    control_fn = (
        make_pid_control(dof=args.dof, timestep=args.timestep)
        if args.control == "pid"
        else proportional_control
    )
    robot = SoftPhysioRobot(
        dof=args.dof,
        control_function=control_fn,
        actuator_limits=np.array([[0, 2]] * args.dof, dtype=float),
        timestep=args.timestep,
    )

    history = robot.perform_exercise(
        robot_inputs,
        exercise_name=f"{args.example}_{args.mode}",
        real_time=False,
    )
    visualize_pose_and_robot(
        pose_seq,
        history,
        mode=args.mode,
        interval=args.interval_ms,
        anthropometric_profile=anthropometric_profile,
    )


# ============================================================
# 🚀 Run
# ============================================================
if __name__ == "__main__":
    main()
