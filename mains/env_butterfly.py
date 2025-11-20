from pettingzoo.butterfly import knights_archers_zombies_v10
import matplotlib.pyplot as plt

env = knights_archers_zombies_v10.env(render_mode="human", vector_state=False, num_archers=2, num_knights=0)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()
    print(reward)
    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()