from mawm.envs.marl_grid import make_env
from mawm.envs.marl_grid.cfg import config
from mawm.envs.marl_grid.wrappers import GridRecorder
import numpy as np

config.env_cfg.seed = np.random.randint(0, 10000)
config.env_cfg.max_steps = 2000
env = make_env(config.env_cfg)

env = GridRecorder(env, save_root=".", render_kwargs={"tile_size": 11}, max_steps=config.env_cfg.max_steps)
env.recording = True

class Action: 
    right = 0 # Move right 
    down = 1 # Move down 
    left = 2 # Move left 
    up = 3 # Move up 
    done = 4 # Done completing task / Stay 
action_mapping = {
    0: "Right",
    1: "Down",
    2: "Left",
    3: "Up",
    4: "Done",
    }



agents = [f'agent_{i}' for i in range(config.env_cfg.num_agents)]

lst_actions = [] 
lst_dirs = []
done = False
step = 0

import numpy as np


obs = env.reset()
goal_pos = obs['global']['goal_pos']
goals  = {}
for i, agent in enumerate(agents):
    goal =  env.get_goal(agent= env.agents[i], goal_pos= (goal_pos[0], goal_pos[1]))
    goals[agent] = goal


lst_dirs.append({f"agent_{i}": (agent.dir, agent.pos) for i, agent in enumerate(env.agents)})

while step < 100:
    actions = {agent: None for agent in agents}
    for i, agent in enumerate(agents):
        action = env.action_space.sample()
        actions[agent] = action
    
    print(f"Step {step}, Actions: {action_mapping[actions[agents[0]]]}, {action_mapping[actions[agents[1]]]}")
    
    obs, rewards, done, infos = env.step(actions)  # Execute FIRST
    
    lst_actions.append({agent: action_mapping[action] for agent, action in actions.items()})
    lst_dirs.append({f"agent_{i}": (agent.dir, agent.pos) for i, agent in enumerate(env.agents)})  # Record AFTER
    
    done = done['__all__']
    step += 1
    
env.export_frames(save_root=".")
np.save("actions.npy", np.array(lst_actions))
np.save("directions.npy", np.array(lst_dirs))
import pickle
with open("goals.pkl", "wb") as f:
    pickle.dump(goals, f)

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









