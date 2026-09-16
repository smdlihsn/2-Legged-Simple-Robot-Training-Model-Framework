import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np
import time

class CrawlerEnv(gym.Env):
    def __init__(self, render_mode=None):
        super(CrawlerEnv, self).__init__()
        
        self.render_mode = render_mode
        
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # FIX: Bumped to shape=(7,) to fit our new forward speedometer!
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.prev_x = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        self.planeId = p.loadURDF("plane.urdf")
        self.robotId = p.loadURDF("crawler_shortleg.urdf", [0, 0, 0.15], p.getQuaternionFromEuler([0, 0, 0]))
        
        self.joint_map = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            self.joint_map[joint_info[1].decode('utf-8')] = i
            
        self.left_idx = self.joint_map['left_servo']
        self.right_idx = self.joint_map['right_servo']
        
        self.prev_x = 0.0
        
        return self._get_obs(), {}

    def _get_obs(self):
        left_pos, _, _, _ = p.getJointState(self.robotId, self.left_idx)
        right_pos, _, _, _ = p.getJointState(self.robotId, self.right_idx)
        
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        linear_vel, ang_vel = p.getBaseVelocity(self.robotId)
        euler = p.getEulerFromQuaternion(ori)
        
        # FIX: Added world_x_velocity (linear_vel[0]) so the AI can feel its own speed!
        obs = np.array([
            left_pos,           # 0
            right_pos,          # 1
            euler[0],           # 2: Roll
            euler[1],           # 3: Pitch
            ang_vel[0],         # 4: Roll velocity
            ang_vel[1],         # 5: Pitch velocity
            linear_vel[0]       # 6: NEW - World X Velocity (Speedometer)
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        # 1. ABSOLUTE TARGETS: The AI explicitly chooses the absolute angle (-90 to +90 degrees)
        # This completely bypasses the stall-feedback loop of Delta Control
        left_target = float(action[0] * 1.57)
        right_target = float(action[1] * 1.57)
        
        # Track current state for the jerkiness check
        current_left = p.getJointState(self.robotId, self.left_idx)[0]
        current_right = p.getJointState(self.robotId, self.right_idx)[0]
        
        # 2. Apply motor commands with proper force
        p.setJointMotorControl2(self.robotId, self.left_idx, p.POSITION_CONTROL, targetPosition=left_target, force=3.5)
        p.setJointMotorControl2(self.robotId, self.right_idx, p.POSITION_CONTROL, targetPosition=right_target, force=3.5)
        
        # 3. Frame skipping loop
        frame_skip = 8
        for _ in range(frame_skip):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)
            
        # 4. Get Velocity and Positions
        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        world_linear_vel, _ = p.getBaseVelocity(self.robotId)
        world_x_speed = world_linear_vel[0]
        
        # 5. REWARD MATH (Clean, simple, non-clamped)
        reward = float(world_x_speed * 100.0)
        
        
        # THE ANTI-JUMPING GUARDRAIL: Heavily penalize massive teleports
        # If the AI commands a massive change from where the leg currently is, it loses points.
        # This forces the AI to move smoothly without using Delta Control!
        jerkiness_penalty = abs(current_left - left_target) + abs(current_right - right_target)
        reward -= jerkiness_penalty * 2.0  # Increased multiplier to penalize jumping harshly
        
        # 6. Check for wipeouts
        obs = self._get_obs()
        roll, pitch = obs[2], obs[3]
        terminated = False

        if abs(roll) > 1.2 or abs(pitch) > 1.2:
            terminated = True
            reward -= 10.0
            
        truncated = False


        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect()