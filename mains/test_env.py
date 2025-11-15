# from MAWM.envs.marl_grid.envs.findgoal_env.environments import make_environment
# from MAWM.envs.marl_grid.envs.findgoal_env.cfgs import config

# create_env = lambda: make_environment(config.env_cfg)

# env = create_env()
# # agents = [f'agent_{i}' for i in range(env.num_agents)]
# # obs = env.reset()
# # print(obs.keys())
# # img = obs['agent_0']['pov']
# # print(img.shape)
# # print(obs['agent_0'].keys())
# # import matplotlib.pyplot as plt
# # plt.imshow(img)
# # plt.show()







# # Reset environment to start a new episode
# # obs_dict = env.reset()
# # print(obs_dict.keys())
# # print(obs_dict['agent_0'].keys())
# # # print(obs_dict['global'].keys())
# # print()
# # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
# # info: extra debugging information (usually not needed for basic learning)

# # print(f"Starting observation: {observation}")
# # # Example output: [ 0.01234567 -0.00987654  0.02345678  0.01456789]
# # # [cart_position, cart_velocity, pole_angle, pole_angular_velocity]

# # episode_over = False
# # total_reward = 0

# # while not episode_over:
# #     # Choose an action: 0 = push cart left, 1 = push cart right
# #     action = env.action_space.sample()  # Random action for now - real agents will be smarter!
# #     import pdb; pdb.set_trace()
# #     # Take the action and see what happens
# #     observation, reward, terminated, truncated, info = env.step(action)

# #     # reward: +1 for each step the pole stays upright
# #     # terminated: True if pole falls too far (agent failed)
# #     # truncated: True if we hit the time limit (500 steps)

# #     total_reward += reward
# #     episode_over = terminated or truncated

# # print(f"Episode finished! Total reward: {total_reward}")


# import torch
# import numpy as np
# agents = [f'agent_{i}' for i in range(config.env_cfg.num_agents)]


# trajectories = [ [] for _ in agents ]  # list per agent

# # num_episodes = 100
# obs = env.reset()


# done = False
# step = 0
# # loop while numpy array 'done' is not all True
# while not done and step < config.env_cfg.max_steps:
#     # For each agent: encode obs, generate message, choose action
#     messages = []
#     actions = {agent: None for agent in agents}
#     latents = []
#     for i, agent in enumerate(agents):
#         # import pdb; pdb.set_trace()
#         o = obs[agent]['pov']  # or however obs is structured
#         action = env.action_space.sample()  # placeholder for agent action
#         actions[agent] = action
#     print(obs['global'].keys())
#     # Step in environment
#     next_obs, rewards, done, infos = env.step(actions)  # depends on their env API
#     # print(f"Step {step}: rewards = {rewards}, done = {done}")
#     print(done['__all__'])
#     done = done['__all__']
#     # print(np.all(done))
#     # # Save transitions
#     # for i, agent in enumerate(agents):
#     #     trajectories[i].append({
#     #         "obs": obs[agent]['pov'],
#     #         "action": actions[agent],
#     #         "reward": rewards[agent],
#     #         "next_obs": next_obs[agent]['pov'],
#     #     })

#     # obs = next_obs
#     step += 1

# import matplotlib.pyplot as plt
# img = o#next_obs['agent_1']['pov']
# plt.imshow(img) 
# plt.show()
# print("Episode finished!")
# env.close()



















# # # Create environment

# # # Create your agent(s)
# # # They will have image encoder, message encoder / decoder, policy, etc.

# # # Data buffer


# #     # Reset any agent internal state (hidden layers, etc.)
# #     # for agent in agents:
# #     #     agent.reset_internal_state()

# #     while not done and step < config.env_cfg.max_steps:
# #         # For each agent: encode obs, generate message, choose action
# #         messages = []
# #         actions = []
# #         latents = []
# #         for i, agent in enumerate(agents):
# #             import pdb; pdb.set_trace()
# #             o = obs[agent]['pov']  # or however obs is structured
# #             action = env.action_space.sample()  # placeholder for agent action
# #             # latent = agent.encode(o)  # autoencoder
# #             # msg = agent.message_from_latent(latent)
# #             # action = agent.policy(latent, msg, received_messages=None)  # or however their policy works
# #             # latents.append(latent)
# #             # messages.append(msg)
# #             actions.append(action)

# #         # Step in environment
# #         next_obs, rewards, done, infos = env.step(actions, messages)  # depends on their env API
# #         # (Note: env.step might or might not take messages — check code!)
# #         # obs_dict, rew_dict, done_dict, info_dict

# #         # Save transitions
# #         for i, agent in enumerate(agents):
# #             trajectories[i].append({
# #                 "obs": obs[agent]['pov'],
# #                 # "latent": latents[i].detach().cpu().numpy(),
# #                 # "msg": messages[i].detach().cpu().numpy(),
# #                 "action": actions[agent].detach().cpu().numpy() if isinstance(actions[i], torch.Tensor) else actions[i],
# #                 "reward": rewards[agent],
# #                 "next_obs": next_obs[agent]['pov'],
# #                 # "done": done,
# #                 # optionally: "info": infos[i]
# #             })

# #         obs = next_obs
# #         step += 1

# #     # # (optionally) when episode ends, dump / process trajectories
# #     # for i in range(len(agents)):
# #     #     # e.g., save to file
# #     #     import pickle
# #     #     with open(f"agent_{i}_traj_ep{ep}.pkl", "wb") as f:
# #     #         pickle.dump(trajectories[i], f)

# #     #     # reset for next episode
# #     #     trajectories[i] = []



# # env.close()