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
    load_anthropometric_profile,
    scale_pose_sequence,
    select_best_source_index,
)
from visualize_robot_arm import forward_kinematics_points

def visualize_shoulder_mapping(mode="3d", example="Ex1", idx=0, config_file="test_config.json", 
                              max_frames=50, fps=10, autoplay=True, save_gif=False,
                              auto_select_source=False, max_source_candidates=None):
    """Visualize shoulder mapping with pose + DOF charts."""
    
    # Load config
    with open(config_file, "r") as f:
        cfg = json.load(f)
    
    # Load comprehensive anthropometric profile
    anthropometric_profile = load_anthropometric_profile(cfg)
    print("📏 Anthropometric Profile Loaded:")
    anthropometric_profile.print_profile()
    
    # Optionally auto-select best source subject by anthropometric similarity
    if auto_select_source:
        try:
            best_idx, ffiles, dists, inferred_list = select_best_source_index(
                ".", example, mode, anthropometric_profile, max_candidates=max_source_candidates
            )
            print("\n🔎 Auto-selected source sequence for visualization:")
            print(f"  Example: {example}")
            print(f"  Candidates: {len(ffiles)} | Considered: {len(dists)}")
            print(f"  Selected index: {best_idx} | File: {ffiles[best_idx] if best_idx < len(ffiles) else 'N/A'}")
            print(f"  Distance: {dists[best_idx]:.3f}")
            inf = inferred_list[best_idx]
            if inf is not None:
                print("  Inferred (in):", {k: round(v, 2) for k, v in inf.items() if np.isfinite(v)})
            idx = best_idx
        except Exception as e:
            print(f"[warn] Auto-selection failed in visualizer: {e}. Using provided idx={idx}")

    # Load pose data (with possibly updated idx)
    pose_seq_raw = load_pose_sequence(".", idx, example, mode, max_frames=max_frames)
    print(f"📊 Loaded {pose_seq_raw.shape[0]} frames")
    
    # Apply anthropometric scaling for personalized transposition (create adjusted copy)
    print(f"🔧 Applying anthropometric scaling for personalized transposition...")
    pose_seq_adj = scale_pose_sequence(pose_seq_raw, anthropometric_profile, mode)
    print(f"✅ Pose sequence scaled using personal measurements")
    
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
    
    # Generate DOF mapping (from adjusted sequence)
    if mode == "3d":
        dofs = map_shoulders_to_dofs_3d_sequence(
            pose_seq_adj,
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
            pose_seq_adj,
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

    # Compute right-arm robot joint angles (deg) if 3D (from adjusted sequence)
    robot_angles = None
    if mode == "3d":
        robot_angles = compute_right_arm_robot_angles_3d_sequence(
            pose_seq_adj,
            shoulder_idx=joints.get("r_sh_idx", 11),
            elbow_idx=joints.get("r_el_idx", 13),
            wrist_idx=14,
        )
    
    # Create visualization
    connections = get_joint_connections()
    num_frames = min(len(dofs), pose_seq_raw.shape[0], pose_seq_adj.shape[0])
    
    # Get scaling factors for display
    factors = anthropometric_profile.get_limb_scaling_factors()
    scaling_info = f"Scaling: H:{factors['overall']:.2f}x, L-Arm:{factors['left_arm']:.2f}x, R-Arm:{factors['right_arm']:.2f}x"
    
    fig = plt.figure(figsize=(22, 10))
    fig.suptitle(f"Shoulder Mapping Visualization - {scaling_info}", fontsize=12)
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    # 2 rows x 6 cols: top has three equal panels (each spans 2 cols); bottom has 4 mini charts (first 4 cols)
    gs = GridSpec(2, 6, figure=fig, height_ratios=[2.0, 1.0], wspace=0.25, hspace=0.60)

    # Top row (full width split into two halves): original and adjusted poses
    if mode == "3d":
        ax_pose_orig = fig.add_subplot(gs[0, 0:2], projection='3d')
        ax_pose_adj = fig.add_subplot(gs[0, 2:4], projection='3d')
        ax_robot = fig.add_subplot(gs[0, 4:6], projection='3d')
    else:
        ax_pose_orig = fig.add_subplot(gs[0, 0:2])
        ax_pose_adj = fig.add_subplot(gs[0, 2:4])
        ax_robot = fig.add_subplot(gs[0, 4:6])

    # Bottom row: create a sub-gridspec spanning full width with 4 equal columns
    bottom = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1, 0:6], wspace=0.35)
    ax_th0 = fig.add_subplot(bottom[0, 0])
    ax_th1 = fig.add_subplot(bottom[0, 1])
    ax_th2 = fig.add_subplot(bottom[0, 2])
    ax_th3 = fig.add_subplot(bottom[0, 3])

    # Draw a thin labeled separator just above the bottom row
    try:
        # Place separator further above the bottom row (~1–2 cm visual gap depending on DPI)
        y_sep = ax_th0.get_position().y1 + 0.05
        fig.add_artist(plt.Line2D([0.05, 0.95], [y_sep, y_sep], transform=fig.transFigure, color='0.5', linewidth=1))
        fig.text(0.5, y_sep + 0.01, "Robot Angles (deg)", ha='center', va='bottom', fontsize=10, color='0.5', transform=fig.transFigure)
    except Exception:
        pass
    
    # (Time series removed for compact 2x4 layout)
    
    # Precompute fixed axes limits across all frames to make size differences visible
    fixed_limits = None
    try:
        if mode == "3d":
            xs, ys, zs = [], [], []
            sample_stride = max(1, num_frames // 100)
            for t in range(0, num_frames, sample_stride):
                # Use both raw and adjusted to span full range
                for seq in (pose_seq_raw, pose_seq_adj):
                    pts = None
                    try:
                        pts = seq[t]
                    except Exception:
                        continue
                    pts3 = None
                    try:
                        if pts.ndim == 2 and pts.shape[1] >= 3:
                            pts3 = pts[:, :3]
                        elif pts.ndim == 2 and pts.shape[0] >= 3:
                            pts3 = pts[:3, :].T
                        else:
                            flat = pts.ravel()
                            if flat.size % 3 == 0:
                                pts3 = flat.reshape(-1, 3)
                            elif flat.size % 4 == 0:
                                pts3 = flat.reshape(-1, 4)[:, :3]
                    except Exception:
                        pts3 = None
                    if pts3 is None:
                        continue
                    xs.append(pts3[:,0])
                    ys.append(pts3[:,1])
                    zs.append(pts3[:,2])
            if xs and ys and zs:
                x_all = np.concatenate(xs); y_all = np.concatenate(ys); z_all = np.concatenate(zs)
                pad = 0.1
                fixed_limits = {
                    "x": (float(x_all.min()-pad), float(x_all.max()+pad)),
                    "y": (float(y_all.min()-pad), float(y_all.max()+pad)),
                    "z": (float(z_all.min()-pad), float(z_all.max()+pad)),
                }
        else:
            xs, ys = [], []
            sample_stride = max(1, num_frames // 100)
            for t in range(0, num_frames, sample_stride):
                for seq in (pose_seq_raw, pose_seq_adj):
                    pts = None
                    try:
                        pts = seq[t]
                    except Exception:
                        continue
                    pts2 = None
                    try:
                        if pts.ndim == 2 and pts.shape[1] >= 2:
                            pts2 = pts[:, :2]
                        elif pts.ndim == 2 and pts.shape[0] >= 2:
                            pts2 = pts[:2, :].T
                        else:
                            flat = pts.ravel()
                            if flat.size % 2 == 0:
                                pts2 = flat.reshape(-1, 2)
                            elif flat.size % 3 == 0:
                                pts2 = flat.reshape(-1, 3)[:, :2]
                    except Exception:
                        pts2 = None
                    if pts2 is None:
                        continue
                    xs.append(pts2[:,0]); ys.append(pts2[:,1])
            if xs and ys:
                x_all = np.concatenate(xs); y_all = np.concatenate(ys)
                pad = 50
                fixed_limits = {
                    "x": (float(x_all.min()-pad), float(x_all.max()+pad)),
                    # For image coords, y is inverted in plotting section
                    "y": (float(y_all.max()+pad), float(y_all.min()-pad)),
                }
    except Exception:
        fixed_limits = None

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

    def get_original_pose_for_comparison(frame_arr, mode):
        """Get original pose data for comparison (before anthropometric scaling)."""
        # This would need to be stored separately if we want to show original vs scaled
        # For now, we'll just return the current frame as both original and scaled
        if mode == "2d":
            return _to2d(frame_arr)
        else:
            return _to3d(frame_arr)

    def animate_frame(t):
        # Clear all axes
        ax_pose_orig.clear()
        ax_pose_adj.clear()
        ax_robot.clear()
        ax_th0.clear()
        ax_th1.clear()
        ax_th2.clear()
        ax_th3.clear()
        
        # Plot pose: left = ground truth (raw), right = adjusted (scaled)
        if mode == "3d":
            pts3_raw = _to3d(pose_seq_raw[t])
            pts3_adj = _to3d(pose_seq_adj[t])
            if pts3_raw is not None and pts3_adj is not None:
                # Ground truth (left)
                x, y, z = pts3_raw[:, 0], pts3_raw[:, 1], pts3_raw[:, 2]
                ax_pose_orig.scatter(x, y, z, c='#d62728', s=25, alpha=0.9)  # red vertices
                for sidx, eidx in connections:
                    if sidx < len(pts3_raw) and eidx < len(pts3_raw):
                        ax_pose_orig.plot([x[sidx], x[eidx]], [y[sidx], y[eidx]], [z[sidx], z[eidx]], color='#1f77b4', lw=1.5)  # blue edges
                ax_pose_orig.set_title(f"Ground Truth 3D (t={t})")

                # Adjusted (right)
                xa, ya, za = pts3_adj[:, 0], pts3_adj[:, 1], pts3_adj[:, 2]
                ax_pose_adj.scatter(xa, ya, za, c='#2ca02c', s=30, alpha=0.9)  # green vertices
                for sidx, eidx in connections:
                    if sidx < len(pts3_adj) and eidx < len(pts3_adj):
                        ax_pose_adj.plot([xa[sidx], xa[eidx]], [ya[sidx], ya[eidx]], [za[sidx], za[eidx]], color='#17becf', lw=1.5)  # cyan edges
                ax_pose_adj.set_title("Adjusted 3D (personalized)")
                
                # Apply fixed limits if available; otherwise sync to combined extents
                if fixed_limits and all(k in fixed_limits for k in ("x","y","z")):
                    ax_pose_orig.set_xlim(*fixed_limits["x"])
                    ax_pose_orig.set_ylim(*fixed_limits["y"])
                    ax_pose_orig.set_zlim(*fixed_limits["z"])
                    ax_pose_adj.set_xlim(*fixed_limits["x"])
                    ax_pose_adj.set_ylim(*fixed_limits["y"])
                    ax_pose_adj.set_zlim(*fixed_limits["z"])
                else:
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
                ax_pose_orig.text(0.5, 0.5, f"Frame {t}\n(3D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_orig.transAxes)
                ax_pose_adj.text(0.5, 0.5, f"Frame {t}\n(3D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_adj.transAxes)
        else:  # 2d
            pts2_raw = _to2d(pose_seq_raw[t])
            pts2_adj = _to2d(pose_seq_adj[t])
            if pts2_raw is not None and pts2_adj is not None:
                x, y = pts2_raw[:, 0], pts2_raw[:, 1]
                ax_pose_orig.scatter(x, y, c='#d62728', s=30)  # red vertices
                for sidx, eidx in connections:
                    if sidx < len(pts2_raw) and eidx < len(pts2_raw):
                        ax_pose_orig.plot([x[sidx], x[eidx]], [y[sidx], y[eidx]], color='#1f77b4')  # blue edges
                ax_pose_orig.set_title(f"Ground Truth 2D (t={t})")

                xa, ya = pts2_adj[:, 0], pts2_adj[:, 1]
                ax_pose_adj.scatter(xa, ya, c='#2ca02c', s=30)  # green vertices
                for sidx, eidx in connections:
                    if sidx < len(pts2_adj) and eidx < len(pts2_adj):
                        ax_pose_adj.plot([xa[sidx], xa[eidx]], [ya[sidx], ya[eidx]], color='#17becf')  # cyan edges
                ax_pose_adj.set_title("Adjusted 2D (personalized)")
                
                # Apply fixed limits if available; otherwise sync to combined extents (note Y flipped)
                if fixed_limits and all(k in fixed_limits for k in ("x","y")):
                    ax_pose_orig.set_xlim(*fixed_limits["x"])
                    ax_pose_adj.set_xlim(*fixed_limits["x"])
                    ax_pose_orig.set_ylim(*fixed_limits["y"])
                    ax_pose_adj.set_ylim(*fixed_limits["y"])
                else:
                    x_min = min(x.min(), xa.min()); x_max = max(x.max(), xa.max())
                    y_min = min(y.min(), ya.min()); y_max = max(y.max(), ya.max())
                    pad = 50
                    ax_pose_orig.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_adj.set_xlim([x_min - pad, x_max + pad])
                    ax_pose_orig.set_ylim([y_max + pad, y_min - pad])
                    ax_pose_adj.set_ylim([y_max + pad, y_min - pad])
            else:
                ax_pose_orig.text(0.5, 0.5, f"Frame {t}\n(2D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_orig.transAxes)
                ax_pose_adj.text(0.5, 0.5, f"Frame {t}\n(2D data unavailable)", 
                           ha='center', va='center', transform=ax_pose_adj.transAxes)
        
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
            ax_th0.set_yticks(range(-90, 91, 30))
            ax_th0.grid(True, axis='y', alpha=0.2)
            ax_th0.text(0, th0, f"{th0:.0f}°", ha='center', va='bottom' if th0>=0 else 'top', fontsize=9, color='#4d4d4d')
            ax_th0.set_title(f'θ0 rot')
            ax_th0.set_ylabel('deg')
            # theta1 [0, 120]
            ax_th1.bar([0], [th1], color='#8c564b', alpha=0.85)
            ax_th1.set_ylim(0, 120)
            ax_th1.set_xticks([])
            ax_th1.set_yticks(range(0, 121, 30))
            ax_th1.grid(True, axis='y', alpha=0.2)
            ax_th1.text(0, th1, f"{th1:.0f}°", ha='center', va='bottom', fontsize=9, color='#4d4d4d')
            ax_th1.set_title(f'θ1 elev')
            # theta2 [0, 135]
            ax_th2.bar([0], [th2], color='#e377c2', alpha=0.85)
            ax_th2.set_ylim(0, 135)
            ax_th2.set_xticks([])
            ax_th2.set_yticks(range(0, 136, 45))
            ax_th2.grid(True, axis='y', alpha=0.2)
            ax_th2.text(0, th2, f"{th2:.0f}°", ha='center', va='bottom', fontsize=9, color='#4d4d4d')
            ax_th2.set_title(f'θ2 elbow')
            # theta3 [-90, 90]
            ax_th3.bar([0], [th3], color='#7f7f7f', alpha=0.85)
            ax_th3.axhline(0, color='k', linewidth=1)
            ax_th3.set_ylim(-90, 90)
            ax_th3.set_xticks([])
            ax_th3.set_yticks(range(-90, 91, 30))
            ax_th3.grid(True, axis='y', alpha=0.2)
            ax_th3.text(0, th3, f"{th3:.0f}°", ha='center', va='bottom' if th3>=0 else 'top', fontsize=9, color='#4d4d4d')
            ax_th3.set_title(f'θ3 pitch')
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
    print(f"\n📊 DOF Summary (Anthropometrically Scaled):")
    print(f"  L Flex: {dofs[:, 0].min():.3f} - {dofs[:, 0].max():.3f} (mean: {dofs[:, 0].mean():.3f})")
    print(f"  R Flex: {dofs[:, 1].min():.3f} - {dofs[:, 1].max():.3f} (mean: {dofs[:, 1].mean():.3f})")
    print(f"  L Abd:  {dofs[:, 2].min():.3f} - {dofs[:, 2].max():.3f} (mean: {dofs[:, 2].mean():.3f})")
    print(f"  R Abd:  {dofs[:, 3].min():.3f} - {dofs[:, 3].max():.3f} (mean: {dofs[:, 3].mean():.3f})")
    
    # Print anthropometric scaling summary
    print(f"\n🔧 Applied Anthropometric Scaling:")
    factors = anthropometric_profile.get_limb_scaling_factors()
    print(f"  Overall Height: {factors['overall']:.2f}x")
    print(f"  Left Arm: {factors['left_arm']:.2f}x")
    print(f"  Right Arm: {factors['right_arm']:.2f}x")
    print(f"  Left Leg: {factors['left_leg']:.2f}x")
    print(f"  Right Leg: {factors['right_leg']:.2f}x")
    print(f"  Shoulder Width: {factors['shoulder_width']:.2f}x")

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
