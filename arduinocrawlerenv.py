import math
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time
import serial
import sys

# Safely set Windows timer resolution only on Win32
if sys.platform == "win32":
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)

class CrawlerEnv(gym.Env):
    def __init__(self, render_mode=None, use_arduino=False):
        super(CrawlerEnv, self).__init__()
        
        self.render_mode = render_mode
        self.use_arduino = use_arduino
        self.arduino_serial = None
        
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.prev_x = 0.0

        # Only connect to Arduino if explicitly requested (e.g. during live testing)
        if self.use_arduino:
            try:
                self.arduino_serial = serial.Serial('COM5', 115200, timeout=0.01)
                time.sleep(2) # Give Arduino time to reboot safely
                print("Digital Twin connection established on COM5!")
            except Exception as e:
                print(f"Arduino Connection failed: {e}")
                self.arduino_serial = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("crawler_goodlegs.urdf", [0, 0, 0.15], p.getQuaternionFromEuler([0, 0, 0]))
        
        self.joint_map = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            self.joint_map[joint_info[1].decode('utf-8')] = i
            
        self.left_idx = self.joint_map['left_servo']
        self.right_idx = self.joint_map['right_servo']
        
        self.step_count = 0
        pos, _ = p.getBasePositionAndOrientation(self.robotId)
        self.prev_x = pos[0]
        
        return self._get_obs(), {}

    def _get_obs(self):
        left_pos, _, _, _ = p.getJointState(self.robotId, self.left_idx)
        right_pos, _, _, _ = p.getJointState(self.robotId, self.right_idx)
        
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        linear_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(ori)
        
        obs = np.array([
            left_pos,           # 0
            right_pos,          # 1
            euler[0],           # 2: Roll
            euler[1],           # 3: Pitch
            euler[2],           # 4: Yaw
            ang_vel[0],         # 5: Roll velocity
            ang_vel[1],         # 6: Pitch velocity
            linear_vel[0]       # 7: World X Speedometer
        ], dtype=np.float32)
        return obs
    
    def step(self, action):
        left_target = float(action[0] * 1.57)
        right_target = float(action[1] * 1.57)

        p.setJointMotorControl2(self.robotId, self.left_idx, p.POSITION_CONTROL, targetPosition=left_target, force=3.5)
        p.setJointMotorControl2(self.robotId, self.right_idx, p.POSITION_CONTROL, targetPosition=right_target, force=3.5)

        frame_skip = 8
        for _ in range(frame_skip):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)

        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        world_linear_vel, _ = p.getBaseVelocity(self.robotId)
        
        # Compute euler directly from orientation quaternion
        euler = p.getEulerFromQuaternion(ori)
        yaw_angle = euler[2]  

        current_x = pos[0]
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        # --- REWARD SYSTEM ---
        # 1. Primary forward progress reward
        reward = delta_x * 200.0

        # 2. Penalty for standing still / zero progress
        if abs(delta_x) < 0.001:
            reward -= 0.5

        # 3. Penalize lateral drift off X-axis and body hopping
        reward -= abs(pos[1]) * 2.0
        reward -= max(0, pos[2] - 0.18) * 10.0

        # 4. Anti-Rotation Penalty (Keep heading straight down X)
        yaw_penalty = abs(yaw_angle) * 15.0
        reward -= yaw_penalty

        # 5. Small contact bonus to encourage pushing off the floor
        left_contact = len(p.getContactPoints(self.robotId, self.planeId, self.left_idx)) > 0
        right_contact = len(p.getContactPoints(self.robotId, self.planeId, self.right_idx)) > 0
        reward += 0.2 * (int(left_contact) + int(right_contact))

        # Check observations & Episode End Conditions
        obs = self._get_obs()
        roll, pitch = obs[2], obs[3]

        self.step_count += 1
        
        # FIX 2: Added Yaw spin-out check to terminations (> 90 deg turn)
        terminated = bool(abs(roll) > 1.2 or abs(pitch) > 1.2 or abs(yaw_angle) > 1.57)
        truncated = bool(self.step_count >= 1000)

        if terminated:
            reward -= 15.0

        # --- SEND DATA TO PHYSICAL TWIN ---
        if self.use_arduino and self.arduino_serial and self.arduino_serial.is_open:
            left_deg = int(math.degrees(obs[0])) + 90
            right_deg = -int(math.degrees(obs[1])) + 90
            
            left_deg = max(0, min(180, left_deg))
            right_deg = max(0, min(180, right_deg))
            
            data_string = f"{left_deg},{right_deg}\n"
            self.arduino_serial.write(data_string.encode('utf-8'))

        return obs, reward, terminated, truncated, {}

    def close(self):
        if self.arduino_serial and self.arduino_serial.is_open:
            self.arduino_serial.close()
            print("Arduino Serial port closed cleanly.")
        p.disconnect()