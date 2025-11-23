from MAWM.envs.marl_grid.envs.findgoal_env.environments import make_environment
from MAWM.envs.marl_grid.envs.findgoal_env.cfgs import config
from MAWM.envs.marl_grid.utils import GridRecorder
import numpy as np
config.env_cfg.seed = 0
config.env_cfg.max_steps = 1000
create_env = lambda: make_environment(config.env_cfg)

env = create_env()
print(env.max_steps)
env = GridRecorder(env, save_root=".", render_kwargs={"tile_size": 11}, max_steps=config.env_cfg.max_steps)




agents = [f'agent_{i}' for i in range(config.env_cfg.num_agents)]



obs = env.reset()
env.recording = True
gx, gy = obs['global']['goal_pos'][0], obs['global']['goal_pos'][1]
# goal_obs = get_goal_observation_before_reaching(env, env.agents[0], (gx, gy))

done = False
step = 0

while not done or step < config.env_cfg.max_steps:

    actions = {agent: None for agent in agents}
    latents = []
    for i, agent in enumerate(agents):
        # import pdb; pdb.set_trace()

        o = obs[agent]['pov']
        action = env.action_space.sample()
        actions[agent] = action
    
    print(obs['global'].keys())
    next_obs, rewards, done, infos = env.step(actions) 
    print(f"Step {step}: rewards = {rewards}, done = {done}")
    done = done['__all__']
    step += 1
 
env.export_video("test_minigrid.mp4")
env.close()














