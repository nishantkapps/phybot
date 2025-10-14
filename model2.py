import numpy as np
import time

class PIDController:
    """
    A configurable PID controller for each degree of freedom.
    """
    def __init__(self, kp, ki, kd, dof, timestep=0.01):
        self.kp = np.full(dof, kp) if np.isscalar(kp) else np.array(kp)
        self.ki = np.full(dof, ki) if np.isscalar(ki) else np.array(ki)
        self.kd = np.full(dof, kd) if np.isscalar(kd) else np.array(kd)
        self.timestep = timestep

        self.integral = np.zeros(dof)
        self.prev_error = np.zeros(dof)

    def compute(self, target, feedback):
        """
        Compute PID output given target (input) and feedback (current state).
        """
        error = target - feedback
        self.integral += error * self.timestep
        derivative = (error - self.prev_error) / self.timestep

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output


class SoftPhysioRobot:
    """
    Configurable soft robot controller for physiotherapy exercises.
    Supports up to 10 DOF and 10 control input parameters.
    """

    def __init__(self, dof, control_function, actuator_limits=None, timestep=0.01):
        self.dof = np.clip(dof, 1, 10)
        self.control_function = control_function
        self.timestep = timestep
        self.state = np.zeros(self.dof)
        self.feedback = np.zeros(self.dof)
        self.actuator_limits = actuator_limits if actuator_limits is not None else np.array([[0, 1]] * self.dof)

    def actuate(self, input_vector):
        """
        Apply control logic to compute actuator commands for each DOF.
        """
        if input_vector.shape[0] > 10:
            raise ValueError("Maximum 10 input parameters allowed.")

        actuator_commands = self.control_function(input_vector, self.feedback, self.dof)

        # Apply actuator limits
        for i in range(self.dof):
            actuator_commands[i] = np.clip(
                actuator_commands[i],
                self.actuator_limits[i, 0],
                self.actuator_limits[i, 1]
            )

        # Simulate next state update (simple dynamics)
        self.state = self.state + actuator_commands * self.timestep
        self.feedback = self.state.copy()

        return actuator_commands, self.state

    def perform_exercise(self, input_sequence, exercise_name="default", real_time=False):
        """
        Execute a physiotherapy exercise sequence.
        """
        print(f"\n🏋️ Starting physiotherapy exercise: {exercise_name}")
        history = []

        for t, inputs in enumerate(input_sequence):
            actuator_cmds, next_state = self.actuate(np.array(inputs))
            history.append({
                "timestep": t,
                "inputs": inputs.tolist(),
                "actuators": actuator_cmds.tolist(),
                "state": next_state.tolist()
            })
            print(f"t={t:02d} | Inputs={inputs} | Actuators={actuator_cmds} | State={next_state}")
            if real_time:
                time.sleep(self.timestep)

        print(f"✅ Completed exercise: {exercise_name}\n")
        return history


# --------------------------
# 🧠 Example Control Strategies
# --------------------------

# Simple proportional control
def proportional_control(inputs, feedback, dof):
    gain = 0.6
    return gain * (inputs[:dof] - feedback[:dof])

# PID-based control (wrapper)
def make_pid_control(kp=0.5, ki=0.05, kd=0.01, dof=6, timestep=0.05):
    pid = PIDController(kp, ki, kd, dof, timestep)
    def pid_control(inputs, feedback, dof):
        return pid.compute(inputs[:dof], feedback[:dof])
    return pid_control


# --------------------------
# 🦿 Predefined Physiotherapy Exercises
# --------------------------

def physiotherapy_exercises(exercise_type, dof):
    """
    Returns a pre-programmed input sequence for a specific exercise.
    Each sequence is simplified to 5 time steps for demonstration.
    """
    base = np.zeros((5, dof))

    if exercise_type.lower() == "wrist_rotation":
        # Sinusoidal wrist movement
        for i in range(5):
            base[i, :] = np.sin(np.linspace(0, np.pi, dof)) * (i+1) * 0.2
    elif exercise_type.lower() == "elbow_flexion":
        # Linear flexion and extension
        base = np.tile(np.linspace(0, 1, dof), (5, 1)) * np.linspace(0.2, 1, 5).reshape(-1, 1)
    elif exercise_type.lower() == "ankle_flexion":
        base = np.abs(np.cos(np.linspace(0, np.pi, dof))) * np.linspace(0.1, 0.9, 5).reshape(-1, 1)
    elif exercise_type.lower() == "shoulder_abduction":
        base = np.tile(np.linspace(0.1, 1, dof), (5, 1)) * np.linspace(0.3, 1, 5).reshape(-1, 1)
    else:
        base = np.random.uniform(0.2, 1.0, (5, dof))  # default random motion

    return base

# ============================================================
# 🧩 Main Function (Entry Point)
# ============================================================
def main():
    print("\n=== Soft Physiotherapy Robot Simulation ===")
    dof = int(input("Enter Degrees of Freedom (1–10): ").strip() or "6")
    control_type = input("Control Type (proportional / pid): ").strip().lower() or "pid"
    exercise_type = input("Exercise Type (wrist_rotation / elbow_flexion / ankle_flexion / shoulder_abduction / random): ").strip().lower() or "elbow_flexion"

    # Select control function
    if control_type == "pid":
        control_fn = make_pid_control(kp=0.8, ki=0.1, kd=0.05, dof=dof, timestep=0.05)
    else:
        control_fn = proportional_control

    # Create robot
    robot = SoftPhysioRobot(
        dof=dof,
        control_function=control_fn,
        actuator_limits=np.array([[0, 2]] * dof),
        timestep=0.05
    )

    # Load exercise pattern
    input_sequence = physiotherapy_exercises(exercise_type, dof)

    # Run the exercise
    robot.perform_exercise(input_sequence, exercise_name=exercise_type, real_time=False)


# ============================================================
# 🚀 Script Entry
# ============================================================
if __name__ == "__main__":
    main()