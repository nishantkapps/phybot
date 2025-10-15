#!/usr/bin/env python3
"""
Robot arm visualizer driven by 4 joint angles in degrees:
  theta0: base/shoulder horizontal rotation (about up)   [-90, +90]
  theta1: shoulder elevation (0=down)                    [0, 120]
  theta2: elbow flexion (0=straight)                     [0, 135]
  theta3: wrist/end-effector pitch (sagittal)            [-90, +90]

Angles are computed per-frame from 3D human pose using model3 utilities,
then rendered as a simple 3-link arm in a torso-fixed frame (up=z,
forward=x, right=y).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from model3 import (
    load_pose_sequence,
    compute_right_arm_robot_angles_3d_sequence,
)


def _rotation_about_axis(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate vector v about unit axis by angle (Rodrigues' formula)."""
    k = axis / (np.linalg.norm(axis) + 1e-12)
    v_par = np.dot(v, k) * k
    v_perp = v - v_par
    v_perp_rot = v_perp * np.cos(angle_rad) + np.cross(k, v_perp) * np.sin(angle_rad)
    return v_par + v_perp_rot


def forward_kinematics_points(theta0, theta1, theta2, theta3, L1=0.321, L2=0.306, Lw=0.10):
    """Return joint points (shoulder, elbow, wrist, hand) in 3D based on angles.

    Conventions:
      - Global axes: up=z, forward=x, right=y
      - theta0: azimuth around +z (up)
      - theta1: elevation from down toward horizontal (0..120 deg)
      - theta2: elbow flexion around local x-axis (hinge), 0=straight
      - theta3: wrist pitch around same hinge axis
    """
    # Global basis
    up = np.array([0.0, 0.0, 1.0])
    forward = np.array([1.0, 0.0, 0.0])
    right = np.array([0.0, 1.0, 0.0])

    # Azimuth direction in horizontal plane
    phi = np.radians(theta0)
    h_dir = np.cos(phi) * forward + np.sin(phi) * right
    h_dir = h_dir / (np.linalg.norm(h_dir) + 1e-12)

    # Shoulder elevation vector: from down (-up) toward horizontal h_dir
    elev = np.radians(theta1)
    d1 = (-np.cos(elev)) * up + (np.sin(elev)) * h_dir
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)

    # Local frame at upper arm for elbow hinge
    z1 = d1
    x1 = np.cross(up, z1)
    if np.linalg.norm(x1) < 1e-9:
        x1 = right.copy()
    x1 = x1 / (np.linalg.norm(x1) + 1e-12)
    y1 = np.cross(z1, x1)
    y1 = y1 / (np.linalg.norm(y1) + 1e-12)

    # Elbow flexion: rotate z1 toward -y1 about x1
    th2 = np.radians(theta2)
    d2 = z1 * np.cos(th2) + (-y1) * np.sin(th2)
    d2 = d2 / (np.linalg.norm(d2) + 1e-12)

    # Wrist pitch: rotate d2 about x1
    th3 = np.radians(theta3)
    d3 = _rotation_about_axis(d2, x1, th3)
    d3 = d3 / (np.linalg.norm(d3) + 1e-12)

    # Points
    P0 = np.zeros(3)           # shoulder/base
    P1 = P0 + L1 * d1          # elbow
    P2 = P1 + L2 * d2          # wrist
    P3 = P2 + Lw * d3          # hand/end-effector
    return P0, P1, P2, P3


def visualize_robot_from_pose(data_dir: str, example: str, idx: int, max_frames: int, fps: int, L1_mm: float = 321.0, L2_mm: float = 306.0, Lw_mm: float = 100.0):
    pose_seq = load_pose_sequence(data_dir, idx, example, mode="3d", max_frames=max_frames)
    angles = compute_right_arm_robot_angles_3d_sequence(pose_seq)
    n = min(len(pose_seq), angles.shape[0])

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    for t in range(n):
        ax.cla()
        th0, th1, th2, th3 = angles[t]
        P0, P1, P2, P3 = forward_kinematics_points(th0, th1, th2, th3, L1=L1_mm/1000.0, L2=L2_mm/1000.0, Lw=Lw_mm/1000.0)

        # Draw links
        X = [P0[0], P1[0], P2[0], P3[0]]
        Y = [P0[1], P1[1], P2[1], P3[1]]
        Z = [P0[2], P1[2], P2[2], P3[2]]
        ax.plot(X[:2], Y[:2], Z[:2], '-o', color='#1f77b4', linewidth=3)
        ax.plot(X[1:3], Y[1:3], Z[1:3], '-o', color='#ff7f0e', linewidth=3)
        ax.plot(X[2:4], Y[2:4], Z[2:4], '-o', color='#2ca02c', linewidth=3)

        # Axes and labels
        ax.set_title(f"Robot Right Arm  t={t} | θ0={th0:.0f}°, θ1={th1:.0f}°, θ2={th2:.0f}°, θ3={th3:.0f}°")
        ax.set_xlabel('X (forward)')
        ax.set_ylabel('Y (right)')
        ax.set_zlabel('Z (up)')
        ax.set_box_aspect((1, 1, 1))
        lim = 0.8
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.1, 1.0)
        ax.view_init(elev=20, azim=-60)
        plt.pause(1.0 / max(1, fps))

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize robot arm from 3D pose-derived joint angles")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--example", type=str, default="Ex1")
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--L1_mm", type=float, default=321.0, help="Upper arm link length (mm)")
    parser.add_argument("--L2_mm", type=float, default=306.0, help="Forearm link length (mm)")
    parser.add_argument("--Lw_mm", type=float, default=100.0, help="Wrist/end effector link length (mm)")
    args = parser.parse_args()

    visualize_robot_from_pose(args.data_dir, args.example, args.idx, args.max_frames, args.fps, args.L1_mm, args.L2_mm, args.Lw_mm)


if __name__ == "__main__":
    main()


