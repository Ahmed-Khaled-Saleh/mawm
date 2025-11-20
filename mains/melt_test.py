# from shimmy import MeltingPotCompatibilityV0
# from shimmy.utils.meltingpot import load_meltingpot

# env = load_meltingpot("collaborative_cooking__circuit")
# env = MeltingPotCompatibilityV0(env, render_mode="human")
# observations = env.reset()

# i =0
# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos = env.step(actions)

#     print(f"Step {i}: rewards = {rewards}, terminations = {terminations}, truncations = {truncations}")    
#     i += 1

# env.close()

from shimmy import MeltingPotCompatibilityV0
from shimmy.utils.meltingpot import load_meltingpot
import pygame

env = load_meltingpot("collaborative_cooking__circuit")
env = MeltingPotCompatibilityV0(env, render_mode="human")

# --- START OF QUICK FIX ---
# The window was created in __init__ with scale=4. We need to:
# 1. Change the scale attribute.
env.display_scale = 16 # Set your desired larger scale (e.g., 8)

# 2. Close the old, small window.
pygame.display.quit()
pygame.quit()

# 3. Re-initialize and create the new, larger window.
pygame.init()
env.clock = pygame.time.Clock()
pygame.display.set_caption("Melting Pot")

shape = env.state_space.shape
env.game_display = pygame.display.set_mode(
    (
        int(shape[1] * env.display_scale),
        int(shape[0] * env.display_scale),
    )
)
# --- END OF QUICK FIX ---

observations = env.reset()

i =0
while env.agents:
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if rewards['player_0'] > 0 or rewards['player_1'] > 0:
        print(rewards)
        print("DONE")
        print("--"*20)
        break
    print(f"Step {i}: rewards = {rewards}, terminations = {terminations}, truncations = {truncations}")    
    i += 1

env.close()