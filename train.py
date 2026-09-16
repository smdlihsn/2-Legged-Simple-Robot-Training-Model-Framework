import gymnasium as gym
from stable_baselines3 import PPO
from arduinocrawlerenv import CrawlerEnv
import os
import time
from stable_baselines3.common.logger import configure

# 1. Create directory to save the trained model
models_dir = "models/PPO"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

print("Initializing environment...")
# Create the environment in DIRECT mode (headless/no graphics) for maximum training speed
env = CrawlerEnv(render_mode="direct")

############# LOGGING #################
log_path = "logs"
new_logger = configure(log_path, ["stdout", "csv"])
if not os.path.exists(log_path):
    os.makedirs(log_path)


policy_kwargs = dict(
    log_std_init=-0.5,          # less aggressive std suppression
    net_arch=dict(pi=[128, 128], vf=[128, 128])  # bigger network
) # Starts the actions out much tighter and controlled


# 2. Initialize the AI Brain (PPO Agent)
# 'MlpPolicy' means a standard Multi-Layer Perceptron neural network (perfect for sensor data)
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    clip_range=0.2,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.01,              # THIS is the key — forces exploration
    max_grad_norm=0.5,          # prevents the giant KL divergence
    policy_kwargs=policy_kwargs,
    device='cpu'
)

# 3. Train the Model
TOTAL_TIMESTEPS = 500000
print(f"Starting training for {TOTAL_TIMESTEPS} steps... dumb robot is trained to crawl")
model.set_logger(new_logger)
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# 4. Save the trained brain
timestamp = time.strftime("%Y%m%d_%H%M%S")

# Save with the timestamp in the name
model.save(f"{models_dir}/crawler_ppo_{timestamp}")
print(f"Brain saved successfully as: crawler_ppo_{timestamp}")
env.close()
print("Training complete! Brain saved.")
