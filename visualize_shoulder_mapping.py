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
    get_joint_connections
)

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
    
    # Create visualization
    connections = get_joint_connections()
    num_frames = min(len(dofs), pose_seq.shape[0])
    
    fig = plt.figure(figsize=(15, 8))
    
    if mode == "3d":
        ax_pose = fig.add_subplot(1, 3, 1, projection='3d')
    else:
        ax_pose = fig.add_subplot(1, 3, 1)
    
    # DOF charts
    ax_l_flex = fig.add_subplot(3, 3, 2)
    ax_r_flex = fig.add_subplot(3, 3, 3)
    ax_l_abd = fig.add_subplot(3, 3, 5)
    ax_r_abd = fig.add_subplot(3, 3, 6)
    ax_time = fig.add_subplot(3, 3, 8)
    
    # Time series of all DOFs
    time_axis = np.arange(num_frames)
    ax_time.plot(time_axis, dofs[:, 0], 'b-', label='L_flex', linewidth=2)
    ax_time.plot(time_axis, dofs[:, 1], 'r-', label='R_flex', linewidth=2)
    ax_time.plot(time_axis, dofs[:, 2], 'b--', label='L_abd', linewidth=2)
    ax_time.plot(time_axis, dofs[:, 3], 'r--', label='R_abd', linewidth=2)
    ax_time.set_xlabel('Frame')
    ax_time.set_ylabel('DOF Value')
    ax_time.set_title('DOF Time Series')
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)
    ax_time.set_ylim(0, 1)
    
    # Animation loop
    def animate_frame(t):
        # Clear all axes
        ax_pose.clear()
        ax_l_flex.clear()
        ax_r_flex.clear()
        ax_l_abd.clear()
        ax_r_abd.clear()
        
        # Plot pose
        if mode == "3d":
            joints = pose_seq[t]
            if joints.ndim == 2 and joints.shape[1] >= 3:
                x, y, z = joints[:, 0], joints[:, 1], joints[:, 2]
                ax_pose.scatter(x, y, z, c='red', s=25, alpha=0.9)
                for s, e in connections:
                    if s < len(joints) and e < len(joints):
                        ax_pose.plot([x[s], x[e]], [y[s], y[e]], [z[s], z[e]], 'b-', lw=1.5)
                ax_pose.set_title(f"3D Pose Frame {t}")
                # Set reasonable limits
                ax_pose.set_xlim([x.min()-0.1, x.max()+0.1])
                ax_pose.set_ylim([y.min()-0.1, y.max()+0.1])
                ax_pose.set_zlim([z.min()-0.1, z.max()+0.1])
            else:
                ax_pose.text(0.5, 0.5, f"Frame {t}\n(3D data unavailable)", 
                           ha='center', va='center', transform=ax_pose.transAxes)
        else:  # 2d
            joints = pose_seq[t]
            if joints.ndim == 2 and joints.shape[1] >= 2:
                x, y = joints[:, 0], joints[:, 1]
                ax_pose.scatter(x, y, c='red', s=30)
                for s, e in connections:
                    if s < len(joints) and e < len(joints):
                        ax_pose.plot([x[s], x[e]], [y[s], y[e]], 'b-')
                ax_pose.set_title(f"2D Pose Frame {t}")
                ax_pose.set_xlim([x.min()-50, x.max()+50])
                ax_pose.set_ylim([y.max()+50, y.min()-50])  # Flip Y for image coords
            else:
                ax_pose.text(0.5, 0.5, f"Frame {t}\n(2D data unavailable)", 
                           ha='center', va='center', transform=ax_pose.transAxes)
        
        # Plot DOF bars
        current_dofs = dofs[t]
        
        # L/R Flexion
        ax_l_flex.bar(['L_Flex'], [current_dofs[0]], color='blue', alpha=0.7)
        ax_l_flex.set_ylim(0, 1)
        ax_l_flex.set_title(f'L Flex: {current_dofs[0]:.3f}')
        ax_l_flex.set_ylabel('DOF Value')
        
        ax_r_flex.bar(['R_Flex'], [current_dofs[1]], color='red', alpha=0.7)
        ax_r_flex.set_ylim(0, 1)
        ax_r_flex.set_title(f'R Flex: {current_dofs[1]:.3f}')
        ax_r_flex.set_ylabel('DOF Value')
        
        # L/R Abduction
        ax_l_abd.bar(['L_Abd'], [current_dofs[2]], color='blue', alpha=0.7)
        ax_l_abd.set_ylim(0, 1)
        ax_l_abd.set_title(f'L Abd: {current_dofs[2]:.3f}')
        ax_l_abd.set_ylabel('DOF Value')
        
        ax_r_abd.bar(['R_Abd'], [current_dofs[3]], color='red', alpha=0.7)
        ax_r_abd.set_ylim(0, 1)
        ax_r_abd.set_title(f'R Abd: {current_dofs[3]:.3f}')
        ax_r_abd.set_ylabel('DOF Value')
        
        # Highlight current frame in time series
        ax_time.axvline(x=t, color='black', linestyle='--', alpha=0.5)
        
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
