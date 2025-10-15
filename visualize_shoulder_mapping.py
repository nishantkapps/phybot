#!/usr/bin/env python3
"""
Visualizer for shoulder mapping results.
Shows original pose + mapped DOF values side by side.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from model3 import (
    map_shoulders_to_dofs_3d_sequence,
    map_shoulders_to_dofs_2d_sequence,
    load_pose_sequence,
    get_joint_connections,
    inches_to_meters,
    compute_right_arm_robot_angles_3d_sequence,
)
from visualize_robot_arm import forward_kinematics_points

def visualize_shoulder_mapping(mode="3d", example="Ex1", idx=0, config_file="test_config.json", 
                              max_frames=50, fps=10, autoplay=True, save_gif=False):
    """Visualize shoulder mapping with pose + DOF charts."""
    
    # Load config
    with open(config_file, "r") as f:
        cfg = json.load(f)
    
    # Load pose data
    pose_seq = load_pose_sequence(".", idx, example, mode, max_frames=max_frames)
    print(f"📊 Loaded {pose_seq.shape[0]} frames")
    
    # Extract config
    joints = cfg.get("joints", {})
    rom = cfg.get("rom_deg", {})
    anthro = cfg.get("anthropometrics", {})
    smoothing = cfg.get("smoothing", {})
    
    # Get ROM settings
    def get_rom_side(rom_dict, side):
        if side in rom_dict:
            return rom_dict[side]
        return rom_dict
    
    l_flex = get_rom_side(rom.get("shoulder_flexion", {}), "L")
    r_flex = get_rom_side(rom.get("shoulder_flexion", {}), "R")
    l_abd = get_rom_side(rom.get("shoulder_abduction", {}), "L")
    r_abd = get_rom_side(rom.get("shoulder_abduction", {}), "R")
    
    # Generate DOF mapping
    if mode == "3d":
        dofs = map_shoulders_to_dofs_3d_sequence(
            pose_seq,
            l_sh_idx=joints.get("l_sh_idx", 6),
            l_el_idx=joints.get("l_el_idx", 8),
            r_sh_idx=joints.get("r_sh_idx", 11),
            r_el_idx=joints.get("r_el_idx", 13),
            flex_rom_min_deg=min(l_flex["min"], r_flex["min"]),
            flex_rom_max_deg=max(l_flex["max"], r_flex["max"]),
            abd_rom_min_deg=min(l_abd["min"], r_abd["min"]),
            abd_rom_max_deg=max(l_abd["max"], r_abd["max"]),
            l_flex_neutral_deg=l_flex["neutral"],
            r_flex_neutral_deg=r_flex["neutral"],
            l_abd_neutral_deg=l_abd["neutral"],
            r_abd_neutral_deg=r_abd["neutral"],
            smoothing_alpha=smoothing.get("alpha", 0.3)
        )
    else:  # 2d
        dofs = map_shoulders_to_dofs_2d_sequence(
            pose_seq,
            l_sh_idx=joints.get("l_sh_idx", 6),
            l_el_idx=joints.get("l_el_idx", 8),
            r_sh_idx=joints.get("r_sh_idx", 11),
            r_el_idx=joints.get("r_el_idx", 13),
            shoulder_width_in=anthro.get("shoulder_width_in", 15.0),
            flex_rom_min_deg=min(l_flex["min"], r_flex["min"]),
            flex_rom_max_deg=max(l_flex["max"], r_flex["max"]),
            abd_rom_min_deg=min(l_abd["min"], r_abd["min"]),
            abd_rom_max_deg=max(l_abd["max"], r_abd["max"]),
            l_flex_neutral_deg=l_flex["neutral"],
            r_flex_neutral_deg=r_flex["neutral"],
            l_abd_neutral_deg=l_abd["neutral"],
            r_abd_neutral_deg=r_abd["neutral"],
            smoothing_alpha=smoothing.get("alpha", 0.3)
        )
    
    print(f"✅ Generated {dofs.shape[0]} DOF frames")

    # Compute right-arm robot joint angles (deg) if 3D
    robot_angles = None
    if mode == "3d":
        robot_angles = compute_right_arm_robot_angles_3d_sequence(
            pose_seq,
            shoulder_idx=joints.get("r_sh_idx", 11),
            elbow_idx=joints.get("r_el_idx", 13),
            wrist_idx=14,
        )
    
    # Create visualization
    connections = get_joint_connections()
    num_frames = min(len(dofs), pose_seq.shape[0])
    
    fig = plt.figure(figsize=(22, 10))
    from matplotlib.gridspec import GridSpec
    # 2 rows x 6 cols: top has three equal panels (each spans 2 cols); bottom has 4 mini charts (first 4 cols)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[2.0, 1.0], wspace=0.25, hspace=0.35)

    # Top row (full width split into two halves): original and adjusted poses
    if mode == "3d":
        ax_pose_orig = fig.add_subplot(gs[0, 0:2], projection='3d')
        ax_pose_adj = fig.add_subplot(gs[0, 2:4], projection='3d')
        ax_robot = fig.add_subplot(gs[0, 4:6], projection='3d')
    else:
        ax_pose_orig = fig.add_subplot(gs[0, 0:2])
        ax_pose_adj = fig.add_subplot(gs[0, 2:4])
        ax_robot = fig.add_subplot(gs[0, 4:6])

    # Bottom row: four compact charts for robot angles (theta0..theta3)
    ax_th0 = fig.add_subplot(gs[1, 0])
    ax_th1 = fig.add_subplot(gs[1, 1])
    ax_th2 = fig.add_subplot(gs[1, 2])
    ax_th3 = fig.add_subplot(gs[1, 3])
    
    # (Time series removed for compact 2x4 layout)
    
    # Animation loop
    def _to2d(arr):
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return arr[:, :2]
        if arr.ndim == 2 and arr.shape[0] >= 2:
            return arr[:2, :].T
        flat = arr.ravel()
        if flat.size % 2 == 0:
            return flat.reshape(-1, 2)
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)[:, :2]
        return None

    def _to3d(arr):
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return arr[:, :3]
        if arr.ndim == 2 and arr.shape[0] >= 3:
            return arr[:3, :].T
        flat = arr.ravel()
        if flat.size % 3 == 0:
            return flat.reshape(-1, 3)
        if flat.size % 4 == 0:
            return flat.reshape(-1, 4)[:, :3]
        return None

    def adjust_pose_by_shoulder_width(frame_arr, mode, shoulder_width_in):
        # Return adjusted joints array with scaling so shoulder width matches provided inches
        if mode == "2d":
            pts = _to2d(frame_arr)
            if pts is None or max(7, 12) >= pts.shape[0]:
                return None
            # Use LeftArm (7) to RightArm (12) span as user-requested shoulder span
            l_sh, r_sh = pts[7], pts[12]
            mid = (l_sh + r_sh) / 2.0
            current = np.linalg.norm(r_sh - l_sh)
            if current <= 1e-6:
                return None
            target_px = shoulder_width_in * 10.0  # 10 px per inch for display
            s = target_px / current
            adj = (pts - mid) * s + mid
            return adj
        else:
            pts = _to3d(frame_arr)
            if pts is None or max(7, 12) >= pts.shape[0]:
                return None
            # Use LeftArm (7) to RightArm (12)
            l_sh, r_sh = pts[7], pts[12]
            mid = (l_sh + r_sh) / 2.0
            current = np.linalg.norm(r_sh - l_sh)
            if current <= 1e-9:
                return None
            target_m = inches_to_meters(shoulder_width_in)
            s = target_m / current
            adj = (pts - mid) * s + mid
            return adj

    def animate_frame(t):
        # Clear all axes
        ax_pose_orig.clear()
        ax_pose_adj.clear()
        ax_robot.clear()
        ax_th0.clear()
        ax_th1.clear()
        ax_th2.clear()
        ax_th3.clear()
        
        # Plot pose: original and adjusted
        if mode == "3d":
            joints = pose_seq[t]
            pts3 = _to3d(joints)
            if pts3 is not None:
                x, y, z = pts3[:, 0], pts3[:, 1], pts3[:, 2]
                ax_pose_orig.scatter(x, y, z, c='#d62728', s=25, alpha=0.9)  # red vertices
                for sidx, eidx in connections:
                    if sidx < len(pts3) and eidx < len(pts3):
                        ax_pose_orig.plot([x[sidx], x[eidx]], [y[sidx], y[eidx]], [z[sidx], z[eidx]], color='#1f77b4', lw=1.5)  # blue edges
                ax_pose_orig.set_title(f"Original 3D (t={t})")

                shoulder_width_in = cfg.get("anthropometrics", {}).get("shoulder_width_in", 15.0)
                adj3 = adjust_pose_by_shoulder_width(pts3, "3d", shoulder_width_in)
                if adj3 is not None:
                    xa, ya, za = adj3[:, 0], adj3[:, 1], adj3[:, 2]
                    ax_pose_adj.scatter(xa, ya, za, c='#2ca02c', s=30, alpha=0.9)  # green vertices
                    for sidx, eidx in connections:
                        if sidx < len(adj3) and eidx < len(adj3):
                            ax_pose_adj.plot([xa[sidx], xa[eidx]], [ya[sidx], ya[eidx]], [za[sidx], za[eidx]], color='#17becf', lw=1.5)  # cyan edges
                    ax_pose_adj.set_title("Adjusted 3D (shoulder width)")
                    # Use shared axis limits for easy comparison
                    x_min = min(x.min(), xa.min()); x_max = max(x.max(), xa.max())
                    y_min = min(y.min(), ya.min()); y_max = max(y.max(), ya.max())
                    z_min = min(z.min(), za.min()); z_max = max(z.max(), za.max())
                    pad = 0.1
                    ax_pose_orig.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_orig.set_ylim([y_min - pad, y_max + pad])
                    ax_pose_orig.set_zlim([z_min - pad, z_max + pad])
                    ax_pose_adj.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_adj.set_ylim([y_min - pad, y_max + pad])
                    ax_pose_adj.set_zlim([z_min - pad, z_max + pad])
                else:
                    # Only original available; apply limits to both for consistency
                    pad = 0.1
                    ax_pose_orig.set_xlim([x.min()-pad, x.max()+pad])
                    ax_pose_orig.set_ylim([y.min()-pad, y.max()+pad])
                    ax_pose_orig.set_zlim([z.min()-pad, z.max()+pad])
                    ax_pose_adj.set_xlim([x.min()-pad, x.max()+pad])
                    ax_pose_adj.set_ylim([y.min()-pad, y.max()+pad])
                    ax_pose_adj.set_zlim([z.min()-pad, z.max()+pad])
            else:
                ax_pose_orig.text(0.5, 0.5, f"Frame {t}\n(3D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_orig.transAxes)
        else:  # 2d
            joints = pose_seq[t]
            pts2 = _to2d(joints)
            if pts2 is not None:
                x, y = pts2[:, 0], pts2[:, 1]
                ax_pose_orig.scatter(x, y, c='#d62728', s=30)  # red vertices
                for sidx, eidx in connections:
                    if sidx < len(pts2) and eidx < len(pts2):
                        ax_pose_orig.plot([x[sidx], x[eidx]], [y[sidx], y[eidx]], color='#1f77b4')  # blue edges
                ax_pose_orig.set_title(f"Original 2D (t={t})")

                shoulder_width_in = cfg.get("anthropometrics", {}).get("shoulder_width_in", 15.0)
                adj2 = adjust_pose_by_shoulder_width(pts2, "2d", shoulder_width_in)
                if adj2 is not None:
                    xa, ya = adj2[:, 0], adj2[:, 1]
                    ax_pose_adj.scatter(xa, ya, c='#2ca02c', s=30)  # green vertices
                    for sidx, eidx in connections:
                        if sidx < len(adj2) and eidx < len(adj2):
                            ax_pose_adj.plot([xa[sidx], xa[eidx]], [ya[sidx], ya[eidx]], color='#17becf')  # cyan edges
                    ax_pose_adj.set_title("Adjusted 2D (shoulder width)")
                    # Shared axis limits for comparison (note Y flipped for image coords)
                    x_min = min(x.min(), xa.min()); x_max = max(x.max(), xa.max())
                    y_min = min(y.min(), ya.min()); y_max = max(y.max(), ya.max())
                    pad = 50
                    ax_pose_orig.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_adj.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_orig.set_ylim([y_max + pad, y_min - pad])
                    ax_pose_adj.set_ylim([y_max + pad, y_min - pad])
                else:
                    # Only original available; apply same limits to both
                    pad = 50
                    ax_pose_orig.set_xlim([x.min()-pad, x.max()+pad])
                    ax_pose_adj.set_xlim([x.min()-pad, x.max()+pad])
                    ax_pose_orig.set_ylim([y.max()+pad, y.min()-pad])
                    ax_pose_adj.set_ylim([y.max()+pad, y.min()-pad])
            else:
                ax_pose_orig.text(0.5, 0.5, f"Frame {t}\n(2D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_orig.transAxes)
        
        # Draw robot pose (only meaningful in 3D mode)
        if robot_angles is not None:
            th0, th1, th2, th3 = robot_angles[min(t, robot_angles.shape[0]-1)]
            # Link lengths (mm) optionally from config
            links = cfg.get("robot_links_mm", {})
            L1_mm = float(links.get("L1_mm", 321.0))
            L2_mm = float(links.get("L2_mm", 306.0))
            Lw_mm = float(links.get("Lw_mm", 100.0))
            P0, P1, P2, P3 = forward_kinematics_points(th0, th1, th2, th3, L1=L1_mm/1000.0, L2=L2_mm/1000.0, Lw=Lw_mm/1000.0)
            X = [P0[0], P1[0], P2[0], P3[0]]
            Y = [P0[1], P1[1], P2[1], P3[1]]
            Z = [P0[2], P1[2], P2[2], P3[2]]
            ax_robot.plot(X[:2], Y[:2], Z[:2], '-o', color='#1f77b4', linewidth=3)
            ax_robot.plot(X[1:3], Y[1:3], Z[1:3], '-o', color='#ff7f0e', linewidth=3)
            ax_robot.plot(X[2:4], Y[2:4], Z[2:4], '-o', color='#2ca02c', linewidth=3)
            ax_robot.set_title(f"Robot (θ0={th0:.0f}°, θ1={th1:.0f}°, θ2={th2:.0f}°, θ3={th3:.0f}°)")
            ax_robot.set_xlabel('X')
            ax_robot.set_ylabel('Y')
            ax_robot.set_zlabel('Z')
            ax_robot.set_box_aspect((1, 1, 1))
            # Synchronize robot axes with original pose panel for side-by-side comparability
            try:
                lx, ux = ax_pose_orig.get_xlim()
                ly, uy = ax_pose_orig.get_ylim()
                lz, uz = ax_pose_orig.get_zlim()
                ax_robot.set_xlim(lx, ux)
                ax_robot.set_ylim(ly, uy)
                ax_robot.set_zlim(lz, uz)
            except Exception:
                # Fallback to default limits
                lim = 0.8
                ax_robot.set_xlim(-lim, lim)
                ax_robot.set_ylim(-lim, lim)
                ax_robot.set_zlim(-0.1, 1.0)
            ax_robot.view_init(elev=20, azim=-60)

        # Plot robot joint angles (deg). If 2D, show N/A.
        if robot_angles is not None:
            th0, th1, th2, th3 = robot_angles[min(t, robot_angles.shape[0]-1)]
            # theta0 [-90, 90]
            ax_th0.bar([0], [th0], color='#9467bd', alpha=0.85)
            ax_th0.axhline(0, color='k', linewidth=1)
            ax_th0.set_ylim(-90, 90)
            ax_th0.set_xticks([])
            ax_th0.set_title(f'θ0 rot {th0:.0f}°')
            ax_th0.set_ylabel('deg')
            # theta1 [0, 120]
            ax_th1.bar([0], [th1], color='#8c564b', alpha=0.85)
            ax_th1.set_ylim(0, 120)
            ax_th1.set_xticks([])
            ax_th1.set_title(f'θ1 elev {th1:.0f}°')
            # theta2 [0, 135]
            ax_th2.bar([0], [th2], color='#e377c2', alpha=0.85)
            ax_th2.set_ylim(0, 135)
            ax_th2.set_xticks([])
            ax_th2.set_title(f'θ2 elbow {th2:.0f}°')
            # theta3 [-90, 90]
            ax_th3.bar([0], [th3], color='#7f7f7f', alpha=0.85)
            ax_th3.axhline(0, color='k', linewidth=1)
            ax_th3.set_ylim(-90, 90)
            ax_th3.set_xticks([])
            ax_th3.set_title(f'θ3 pitch {th3:.0f}°')
        else:
            for ax, name in zip([ax_th0, ax_th1, ax_th2, ax_th3], ['θ0', 'θ1', 'θ2', 'θ3']):
                ax.text(0.5, 0.5, f'{name}\nN/A (2D)', ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
        
        # (No time series panel in this layout)
        
        plt.tight_layout()
    
    # Playback loop
    if autoplay:
        dt = 1.0 / max(1, fps)
        print(f"\n🎬 Autoplay: rendering {num_frames} frames at ~{fps} FPS")
        for t in range(num_frames):
            animate_frame(t)
            plt.pause(dt)
        plt.show()
    else:
        print("\n🎬 Manual mode: press Enter to advance frames")
        for t in range(num_frames):
            animate_frame(t)
            plt.pause(0.01)
            if t < num_frames - 1:
                input("Press Enter for next frame...")
        plt.show()
    
    # Print summary statistics
    print(f"\n📊 DOF Summary:")
    print(f"  L Flex: {dofs[:, 0].min():.3f} - {dofs[:, 0].max():.3f} (mean: {dofs[:, 0].mean():.3f})")
    print(f"  R Flex: {dofs[:, 1].min():.3f} - {dofs[:, 1].max():.3f} (mean: {dofs[:, 1].mean():.3f})")
    print(f"  L Abd:  {dofs[:, 2].min():.3f} - {dofs[:, 2].max():.3f} (mean: {dofs[:, 2].mean():.3f})")
    print(f"  R Abd:  {dofs[:, 3].min():.3f} - {dofs[:, 3].max():.3f} (mean: {dofs[:, 3].mean():.3f})")

def main():
    """Interactive visualizer."""
    print("🎬 Shoulder Mapping Visualizer")
    print("=" * 40)
    
    mode = input("Mode (2d/3d) [3d]: ").strip() or "3d"
    example = input("Example (Ex1/Ex2/...) [Ex1]: ").strip() or "Ex1"
    idx = int(input("Sequence index [0]: ").strip() or "0")
    config_file = input("Config file [test_config.json]: ").strip() or "test_config.json"
    max_frames = int(input("Max frames to process [50]: ").strip() or "50")
    autoplay_in = (input("Autoplay (y/n) [y]: ").strip() or "y").lower()
    autoplay = autoplay_in != "n"
    fps = int(input("Frames per second [10]: ").strip() or "10")
    
    print()
    visualize_shoulder_mapping(mode, example, idx, config_file, max_frames=max_frames, fps=fps, autoplay=autoplay)

if __name__ == "__main__":
    main()
