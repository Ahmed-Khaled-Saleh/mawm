import numpy as np
from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
from stable_baselines3 import PPO
# Import the necessary utilities from SB3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_util import register_for_vec_env

# --- Environment Setup Function (Remains the same) ---
def make_cooking_env_fn(render_mode="rgb_array"):
    """
    Creates and wraps the MeltingPot environment for SB3 compatibility.
    """
    base_env = load_meltingpot("collaborative_cooking__circuit")
    # Note: Using "rgb_array" for training
    wrapped_env = MeltingPotCompatibilityV0(base_env, render_mode=render_mode)
    return wrapped_env

# --- PPO Training Setup ---

# 1. Register the environment creation function
# We register the *function* that returns the wrapped environment.
# This makes it easy to create multiple environments for vectorization later.
# We'll name it "MeltingPotCooking-v0"
register_for_vec_env(
    env_id="MeltingPotCooking-v0",
    env_callable=make_cooking_env_fn,
    env_kwargs={"render_mode": "rgb_array"} # Pass the default training mode
)

# 2. Use make_vec_env to create the vectorized environment (handles DummyVecEnv internally)
# n_envs=1 uses a DummyVecEnv. n_envs > 1 would use SubprocVecEnv.
vec_env = make_vec_env("MeltingPotCooking-v0", n_envs=1, seed=0)

# 3. Initialize the PPO agent
# ... (rest of your code) ...
model = PPO(
    "CnnPolicy", 
    vec_env, 
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    tensorboard_log="./logs/cooking_tensorboard/"
)

# 4. Start Training
TIMESTEPS = 500_000 
print(f"Starting training for {TIMESTEPS} timesteps...")
model.learn(total_timesteps=TIMESTEPS)

model.save("ppo_collaborative_cooking_model")
print("Training complete. Model saved.")