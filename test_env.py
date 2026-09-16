import pybullet as p
import pybullet_data
import time
import math

# 1. Start PyBullet
physicsClient = p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Load Environment and Robot
planeId = p.loadURDF("plane.urdf")
robotStartPos = [0, 0, 0.15]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

robotId = p.loadURDF("crawler.urdf", robotStartPos, robotStartOrientation,)

# 3. Dynamic Joint Lookup (No more hardcoded 0 and 1!)
joint_map = {}
for i in range(p.getNumJoints(robotId)):
    joint_info = p.getJointInfo(robotId, i)
    joint_name = joint_info[1].decode('utf-8')
    joint_map[joint_name] = i

left_servo_idx = joint_map['left_servo']
right_servo_idx = joint_map['right_servo']

print(f"Mapped Joints -> Left: {left_servo_idx}, Right: {right_servo_idx}")

# 4. Simulation Loop
t = 0
while p.isConnected():
    # Since they roll on the Y-axis now, a synchronized pitching motion 
    # should pull the robot forward like a breaststroke swimmer.
    left_target = math.sin(t * 5) 
    right_target = math.sin(t * 5) 
    
    # Send commands using the dynamic indices
    p.setJointMotorControl2(bodyUniqueId=robotId, 
                            jointIndex=left_servo_idx, 
                            controlMode=p.POSITION_CONTROL, 
                            targetPosition=left_target)
                            
    p.setJointMotorControl2(bodyUniqueId=robotId, 
                            jointIndex=right_servo_idx, 
                            controlMode=p.POSITION_CONTROL, 
                            targetPosition=right_target)

    p.stepSimulation()
    
    t += 0.01
    time.sleep(1./240.)