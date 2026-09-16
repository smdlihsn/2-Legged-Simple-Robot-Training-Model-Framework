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
        self.robotId = p.loadURDF("crawler_with_casterball.urdf", [0, 0, 0.15], p.getQuaternionFromEuler([0, 0, 0]))
        
        self.joint_map = {}
        for i in range(p.getNumJoints(self.robotId)):
            joint_info = p.getJointInfo(self.robotId, i)
            self.joint_map[joint_info[1].decode('utf-8')] = i
            
        self.left_idx = self.joint_map['left_servo']
        self.right_idx = self.joint_map['right_servo']
        
        self.step_count = 0
        pos, _ = p.getBasePositionAndOrientation(self.robotId)
        self.prev_x = pos[0]   # <-- from actual position, not hardcoded
        
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
        left_target = float(action[0] * 1.57)
        right_target = float(action[1] * 1.57)

        current_left = p.getJointState(self.robotId, self.left_idx)[0]
        current_right = p.getJointState(self.robotId, self.right_idx)[0]

        p.setJointMotorControl2(self.robotId, self.left_idx, p.POSITION_CONTROL, targetPosition=left_target, force=3.5)
        p.setJointMotorControl2(self.robotId, self.right_idx, p.POSITION_CONTROL, targetPosition=right_target, force=3.5)

        frame_skip = 8
        for _ in range(frame_skip):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)

        pos, ori = p.getBasePositionAndOrientation(self.robotId)
        world_linear_vel, _ = p.getBaseVelocity(self.robotId)

        current_x = pos[0]
        delta_x = current_x - self.prev_x
        self.prev_x = current_x

        # Forward progress
        reward = delta_x * 200.0

        # Punish not moving — threshold raised so it actually bites
        if abs(delta_x) < 0.002:
            reward -= 1.0

        # Penalize sideways drift
        reward -= abs(pos[1]) * 1.5

        # Penalize jumping / leaving the ground
        reward -= abs(world_linear_vel[2]) * 3.0
        reward -= max(0, pos[2] - 0.18) * 10.0

        # Penalize jerkiness lightly — just enough to keep motion smooth
        jerkiness = abs(current_left - left_target) + abs(current_right - right_target)
        reward -= jerkiness * 0.5

        # Reward ground contact — arms should be pushing, not floating
        left_contact = len(p.getContactPoints(self.robotId, self.planeId, self.left_idx)) > 0
        right_contact = len(p.getContactPoints(self.robotId, self.planeId, self.right_idx)) > 0
        reward += 0.3 * (int(left_contact) + int(right_contact))

        # Penalize both arms being frozen simultaneously
        both_still = (abs(left_target - current_left) < 0.05 and
                    abs(right_target - current_right) < 0.05)
        if both_still:
            reward -= 1.0

        obs = self._get_obs()
        roll, pitch = obs[2], obs[3]

        self.step_count += 1
        terminated = bool(abs(roll) > 1.2 or abs(pitch) > 1.2)
        truncated = self.step_count >= 1000

        if terminated:
            reward -= 10.0

        if truncated:
            reward += 5.0  # survived the full episode, good

        return obs, reward, terminated, truncated, {}

    def close(self):
        p.disconnect()