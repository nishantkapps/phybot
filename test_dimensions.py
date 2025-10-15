import numpy as np
from model3 import load_pose_sequence

EXAMPLE = "Ex1"     # change as needed
IDX = 0             # sequence index
MAX_FRAMES = 200    # analyze first N frames

# 19 inches in meters
shoulder_width_user_m = 19 * 0.0254
print("User shoulder width (m):", round(shoulder_width_user_m, 4))

def summarize_widths(widths, label):
    widths = np.array(widths, dtype=float)
    if widths.size == 0:
        print(f"{label}: no data")
        return
    print(f"{label}: frames={len(widths)}, "
          f"min={widths.min():.4f}, max={widths.max():.4f}, mean={widths.mean():.4f}, std={widths.std():.4f}")

# 3D analysis
try:
    pose3d = load_pose_sequence(".", IDX, EXAMPLE, mode="3d", max_frames=MAX_FRAMES)
    widths3d = []
    for f in range(pose3d.shape[0]):
        joints = pose3d[f]
        # Robust slice to [:,3] if extra channels are present
        if joints.ndim == 2 and joints.shape[1] >= 3 and max(7,12) < joints.shape[0]:
            L = joints[7, :3]   # LeftArm (span proxy)
            R = joints[12, :3]  # RightArm (span proxy)
            widths3d.append(np.linalg.norm(R - L))
    summarize_widths(widths3d, "3D shoulder width (data units, likely m)")
    if widths3d:
        ratio = np.array(widths3d) / shoulder_width_user_m
        print(f"3D ratio (data/user): mean={ratio.mean():.3f}, min={ratio.min():.3f}, max={ratio.max():.3f}")
except Exception as e:
    print("3D load/parse failed:", e)

# 2D analysis
try:
    pose2d = load_pose_sequence(".", IDX, EXAMPLE, mode="2d", max_frames=MAX_FRAMES)
    widths2d = []
    for f in range(pose2d.shape[0]):
        joints = pose2d[f]
        if joints.ndim == 2 and joints.shape[1] >= 2 and max(7,12) < joints.shape[0]:
            L = joints[7, :2]   # LeftArm (span proxy)
            R = joints[12, :2]  # RightArm (span proxy)
            widths2d.append(np.linalg.norm(R - L))  # pixels (or whatever 2D units are)
    summarize_widths(widths2d, "2D shoulder width (pixels)")
except Exception as e:
    print("2D load/parse failed:", e)