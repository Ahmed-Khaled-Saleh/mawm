from MAWM.envs.marl_grid.envs.findgoal_env.environments import make_environment
from MAWM.envs.marl_grid.envs.findgoal_env.cfgs import config
config.env_cfg.seed = 42
config.env_cfg.max_steps = 10000
create_env = lambda: make_environment(config.env_cfg)

env = create_env()


agents = [f'agent_{i}' for i in range(config.env_cfg.num_agents)]



obs = env.reset()


done = False
step = 0

while not done and step < config.env_cfg.max_steps:

    actions = {agent: None for agent in agents}
    latents = []
    for i, agent in enumerate(agents):

        o = obs[agent]['pov']
        action = env.action_space.sample()
        actions[agent] = action
    
    print(obs['global'].keys())
    next_obs, rewards, done, infos = env.step(actions) 
    print(f"Step {step}: rewards = {rewards}, done = {done}")
    done = done['__all__']
    step += 1
import pdb; pdb.set_trace()

env.close()



















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