import gymnasium as gym
from stable_baselines3 import PPO
from arduinocrawlerenv import CrawlerEnv
import os

#models_dir = "models/PPO/crawler_ppo_20260613_134950.zip"
#models_dir = "models/PPO/crawler_ppo_20260613_141008.zip"
models_dir = "models\PPO\straigtlinebutfastmovements.zip"

# ==========================================
# EVALUATION: Watch the trained AI crawl!
# ==========================================
print("\nLaunching visual evaluation... Watch your creation!")
eval_env = CrawlerEnv(render_mode="human")

# Load the saved brain
model = PPO.load(f"{models_dir}", env=eval_env, device='cpu')  # Use 'cuda' if you have a compatible GPU

# FIX 1: Get the wrapped vectorized environment from the model.
# This automatically handles the nested array math for us!
vec_env = model.get_env()
obs = vec_env.reset()

print(f"\nRunning infinite playback loop. running the model for {models_dir}")
print("-> To stop it, click on your terminal and press Ctrl + C.")

# FIX 2: Changed to 'while True' so the window stays open forever.
# The vec_env will automatically reset the robot back to the start if it flips over.
while True:
    # Get the smart action from the AI
    action, _states = model.predict(obs, deterministic=True)
    
    # Step the vectorized environment forward
    obs, rewards, dones, infos = vec_env.step(action)

    