import numpy as np
import matplotlib.pyplot as plt
import os
import time
from mpl_toolkits.mplot3d import Axes3D

# === Import functions from your DataLoader ===
from DataLoader import (
    load_video_frames,
    get_xycoords_for_plotting,
    get_joint_connections
)

# ============================================================
# 🧠 PID Controller
# ============================================================
class PIDController:
    def __init__(self, kp, ki, kd, dof, timestep=0.01):
        self.kp = np.full(dof, kp)
        self.ki = np.full(dof, ki)
        self.kd = np.full(dof, kd)
        self.timestep = timestep
        self.integral = np.zeros(dof)
        self.prev_error = np.zeros(dof)

    def compute(self, target, feedback):
        error = target - feedback
        self.integral += error * self.timestep
        derivative = (error - self.prev_error) / self.timestep
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output


# ============================================================
# 🤖 Soft Physiotherapy Robot
# ============================================================
class SoftPhysioRobot:
    def __init__(self, dof, control_function, actuator_limits=None, timestep=0.01):
        self.dof = np.clip(dof, 1, 10)
        self.control_function = control_function
        self.timestep = timestep
        self.state = np.zeros(self.dof)
        self.feedback = np.zeros(self.dof)
        self.actuator_limits = actuator_limits if actuator_limits is not None else np.array([[0, 1]] * self.dof)

    def actuate(self, input_vector):
        actuator_commands = self.control_function(input_vector, self.feedback, self.dof)
        for i in range(self.dof):
            actuator_commands[i] = np.clip(
                actuator_commands[i],
                self.actuator_limits[i, 0],
                self.actuator_limits[i, 1]
            )
        self.state += actuator_commands * self.timestep
        self.feedback = self.state.copy()
        return actuator_commands, self.state

    def perform_exercise(self, input_sequence, exercise_name="default", real_time=False):
        print(f"\n🏋️ Starting physiotherapy exercise: {exercise_name}")
        history = []
        for t, inputs in enumerate(input_sequence):
            actuator_cmds, next_state = self.actuate(np.array(inputs))
            history.append((t, inputs, actuator_cmds, next_state))
            if real_time:
                time.sleep(self.timestep)
        print(f"✅ Completed exercise: {exercise_name}\n")
        return history


# ============================================================
# ⚙️ Control Strategies
# ============================================================
def proportional_control(inputs, feedback, dof):
    gain = 0.6
    return gain * (inputs[:dof] - feedback[:dof])

def make_pid_control(kp=0.8, ki=0.1, kd=0.05, dof=6, timestep=0.05):
    pid = PIDController(kp, ki, kd, dof, timestep)
    def pid_control(inputs, feedback, dof):
        return pid.compute(inputs[:dof], feedback[:dof])
    return pid_control


# ============================================================
# 🏋️ Pose Loader + Mapping to Robot Inputs
# ============================================================
def load_pose_sequence(data_dir, example="Ex1", mode="2d", max_frames=800):
    if mode == "2d":
        path = os.path.join(data_dir, "2d_joints", example)
    else:
        path = os.path.join(data_dir, "3d_joints", example)

    vframes, _ = load_video_frames(path)
    if len(vframes) == 0:
        raise FileNotFoundError(f"No data found for {path}")
 
    return vframes[0][:max_frames]


def convert_pose_to_robot_inputs(pose_sequence, dof):
    frames, joints, dims = pose_sequence.shape
    print('Frames:', frames, ' Joints:', joints, ' Dims:', dims)
    flat_pose = pose_sequence.reshape(frames, joints * dims)
    step = flat_pose.shape[1] // dof
    robot_inputs = np.zeros((frames, dof))
    for i in range(dof):
        robot_inputs[:, i] = np.mean(flat_pose[:, i*step:(i+1)*step], axis=1)
    # Normalize
    robot_inputs = (robot_inputs - robot_inputs.min()) / (robot_inputs.max() + 1e-8)
    return robot_inputs


# ============================================================
# 🎥 Visualization: Pose + Robot State (Side by Side)
# ============================================================
def visualize_pose_and_robot(pose_seq, robot_history, mode="2d", interval=50):
    connections = get_joint_connections()
    num_frames = min(len(robot_history), pose_seq.shape[0])
    dof = robot_history[0][1].shape[0]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax_pose, ax_robot = axes
    plt.ion()

    for t in range(0, num_frames, 2):
        ax_pose.cla()
        ax_robot.cla()

        # Left: Human pose
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
            ax_pose.set_ylim([ymax + 100, ymin - 100])  #
  
        else:
            ax_pose.set_title(f"3D Pose Display (simplified for now)")

        # Right: Robot state
        _, _, act_cmds, next_state = robot_history[t]
        ax_robot.bar(np.arange(dof), next_state, color='orange', alpha=0.7)
        ax_robot.set_ylim(0, 2)
        ax_robot.set_title("Robot Actuator States")
        ax_robot.set_xlabel("DOF Index")
        ax_robot.set_ylabel("State (normalized)")

        plt.pause(interval / 1000)

    plt.ioff()
    plt.show()

def visualize_pose_and_robot(pose_seq, robot_history, mode="2d", interval=50):
    """
    Visualize human pose (2D or 3D) and robot motion side-by-side.
    
    pose_seq: np.array
        - 2D: (num_frames, num_joints, 2)
        - 3D: (num_frames, num_joints, 3)
    robot_history: list of tuples (time, joint_angles)
    mode: "2d" or "3d"
    interval: int
        Pause between frames in milliseconds
    """
    num_frames = min(len(robot_history), len(pose_seq))
    num_joints = pose_seq.shape[1]

    plt.ion()
    fig = plt.figure(figsize=(12, 6))
    
    if mode == "3d":
        ax_pose = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax_pose = fig.add_subplot(1, 2, 1)
    
    ax_robot = fig.add_subplot(1, 2, 2)
    
    # Precompute limits for pose axes
    x_vals = pose_seq[:,:,0].flatten()
    y_vals = pose_seq[:,:,1].flatten()
    if mode == "3d":
        z_vals = pose_seq[:,:,2].flatten()
    
    for t in range(num_frames):
        # --- Plot human pose ---
        ax_pose.cla()
        if mode == "3d":
            ax_pose.scatter(pose_seq[t,:,0], pose_seq[t,:,1], pose_seq[t,:,2], c='r', s=50)
            ax_pose.set_xlim(np.min(x_vals)-0.1, np.max(x_vals)+0.1)
            ax_pose.set_ylim(np.min(y_vals)-0.1, np.max(y_vals)+0.1)
            ax_pose.set_zlim(np.min(z_vals)-0.1, np.max(z_vals)+0.1)
            ax_pose.set_xlabel('X'); ax_pose.set_ylabel('Y'); ax_pose.set_zlabel('Z')
        else:
            ax_pose.scatter(pose_seq[t,:,0], pose_seq[t,:,1], c='r', s=50)
            ax_pose.set_xlim(np.min(x_vals)-0.1, np.max(x_vals)+0.1)
            ax_pose.set_ylim(np.min(y_vals)-0.1, np.max(y_vals)+0.1)
            ax_pose.invert_yaxis()  # optional, for image-like coordinates
        ax_pose.set_title("Human Pose")
        
        # --- Plot robot ---
        ax_robot.cla()
        joint_angles = robot_history[t][1]  # assuming robot_history = list of (time, angles)
        ax_robot.plot(range(len(joint_angles)), joint_angles, 'b.-')
        ax_robot.set_ylim(0, np.max(joint_angles)*1.1)
        ax_robot.set_title("Robot DOF")
        ax_robot.set_xlabel("Joint Index")
        ax_robot.set_ylabel("Angle / Actuator Value")
        
        plt.pause(interval/1000)
    
    plt.ioff()
    plt.show()
# ============================================================
# 🧩 Main Integrated Entry Point
# ============================================================
def main():
    print("\n=== Physiotherapy Robot Simulation using Real Pose Data ===")
    data_dir = input("Enter data directory (default='.'): ").strip() or "."
    example = input("Exercise example (Ex1/Ex2/...): ").strip() or "Ex1"
    mode = input("Data mode (2d/3d): ").strip() or "2d"
    dof = int(input("Robot DOF (1–10): ").strip() or "8")
    control_type = input("Control type (pid/proportional): ").strip() or "pid"

    # 1. Load pose data
    pose_seq = load_pose_sequence(data_dir, example, mode)
    # print('Pose Sequence: ', pose_seq)
    robot_inputs = convert_pose_to_robot_inputs(pose_seq, dof)

    # 2. Create robot + control
    control_fn = make_pid_control(dof=dof, timestep=0.05) if control_type == "pid" else proportional_control
    robot = SoftPhysioRobot(
        dof=dof,
        control_function=control_fn,
        actuator_limits=np.array([[0, 2]] * dof),
        timestep=0.05
    )

    # 3. Run robot motion from pose data
    history = robot.perform_exercise(robot_inputs, exercise_name=f"{example}_{mode}", real_time=False)

    # 4. Visualize human pose + robot motion side-by-side
    visualize_pose_and_robot(pose_seq, history, mode=mode, interval=100)


# ============================================================
# 🚀 Run
# ============================================================
if __name__ == "__main__":
    main()
