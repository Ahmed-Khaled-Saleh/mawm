from mawm.envs.marl_grid import make_env
from mawm.envs.marl_grid.cfg import config
from mawm.envs.marl_grid.wrappers import GridRecorder
import numpy as np

config.env_cfg.seed = np.random.randint(0, 10000)
config.env_cfg.max_steps = 2000
env = make_env(config.env_cfg)

env = GridRecorder(env, save_root=".", render_kwargs={"tile_size": 11}, max_steps=config.env_cfg.max_steps)
env.recording = True

import torch
actions = torch.load("../nbs/models/plan_intents.pkl", map_location=torch.device('cpu'))
print(actions)
