import pybullet as p
import pybullet_data
import serial
import time
import math

# --- SERIAL SETUP ---
# Update 'COM5' to match your actual Arduino port
try:
    # Match the 115200 baud rate from your Arduino sketch
    arduino = serial.Serial('COM5', 115200, timeout=0.01)
    time.sleep(2) # Wait for Arduino reboot
    print("Digital Twin connection established!")
except Exception as e:
    print(f"Connection failed: {e}")
    arduino = None

# --- PYBULLET SETUP ---
physicsClient = p.connect(p.GUI)
p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

planeId = p.loadURDF("plane.urdf")
robotStartPos = [0, 0, 0.15]
robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

robot_id = p.loadURDF("crawler_shortleg.urdf", robotStartPos, robotStartOrientation)
joint_index = 0  # Controlling the first joint

# --- ADD INTERACTIVE GUI SLIDER ---
# Servos usually move -90 to +90 degrees, which is roughly -1.57 to +1.57 radians.
# Parameters: (Slider Name, Minimum Value, Maximum Value, Starting Value)
joint_slider = p.addUserDebugParameter("Servo 1 Position", -1.57, 1.57, 0.0)

# --- MAIN LOOP ---
try:
    while True:
        # 1. Read the current position of the GUI slider (in radians)
        target_radian = p.readUserDebugParameter(joint_slider)
        
        # 2. Tell PyBullet's motor engine to force the joint to that position
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_radian
        )
        
        # 3. Step the physics engine forward
        p.stepSimulation()
        
        # 4. Get the *actual* resulting joint position from the simulation
        joint_state = p.getJointState(robot_id, joint_index)[0]
        
        # 5. Convert radians to degrees
        degrees = int(math.degrees(joint_state))
        
        # 6. Map PyBullet center (0) to physical Servo center (90)
        servo_angle = degrees + 90
        
        # 7. Constrain bounds to prevent hardware damage
        servo_angle = max(0, min(180, servo_angle))
        
        # 8. Send data string to Arduino (e.g., "125\n")
        if arduino and arduino.is_open:
            data_string = f"{servo_angle}\n"
            arduino.write(data_string.encode('utf-8'))
            
        time.sleep(1./240.)

except KeyboardInterrupt:
    if arduino:
        arduino.close()
    p.disconnect()
    print("Digital twin disconnected.")