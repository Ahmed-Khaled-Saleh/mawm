from MAWM.envs.marl_grid.envs.findgoal_env.environments import make_environment
from MAWM.envs.marl_grid.envs.findgoal_env.cfgs import config
from MAWM.envs.marl_grid.utils import GridRecorder
import numpy as np
config.env_cfg.seed = np.random.randint(0, 10000)
config.env_cfg.max_steps = 2000
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

while not done:

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
 
env.export_frames(save_root=".")
env.close()


# import numpy as np

# # read last 100 images from a directory
# dir = "./dir"
# import os
# from PIL import Image
# images = []
# for file_name in sorted(os.listdir(dir)):
#     if file_name.endswith(".png"):
#         img_path = os.path.join(dir, file_name)
#         img = Image.open(img_path)
#         images.append(np.array(img))


# # make a video out of these images
# vido_path = "./res.mp4"
# import cv2
# height, width, layers = images[0].shape
# video = cv2.VideoWriter(vido_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
# for img in images:
#     video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
# video.release()
# print(f"Video saved to {vido_path}")









