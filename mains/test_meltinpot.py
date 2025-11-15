# from shimmy import MeltingPotCompatibilityV0
# from shimmy.utils.meltingpot import load_meltingpot

# env = load_meltingpot("collaborative_cooking__circuit")
# env = MeltingPotCompatibilityV0(env, render_mode="human")
# observations = env.reset()
# lst_agents = []
# i =0

# while env.agents:
#     actions = {agent: env.action_space(agent).sample() for agent in env.agents}
#     observations, rewards, terminations, truncations, infos = env.step(actions)
#     # lst_agents.append(observations['player_0']['RGB'])


#     if i == 10: 
#         break
#     i += 1

# env.close()