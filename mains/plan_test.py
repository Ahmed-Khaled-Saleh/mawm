from mawm.evaluators.planning_eval import PlanEvaluator, preprocessor
from mawm.planners.cem_planner import CEMPlanner

#| hide
from mawm.envs.marl_grid import make_env
from mawm.envs.marl_grid.cfg import config
import copy
import numpy as np

seed = np.random.randint(0, 10000)
cfg = copy.deepcopy(config)
cfg.env_cfg.seed = int(seed)
cfg.env_cfg.max_steps = 512

env = make_env(cfg.env_cfg)


from mawm.models.jepa import JEPA
from omegaconf import OmegaConf
cfg = OmegaConf.load("../cfgs/MPCJepa/mpc.yaml")
model = JEPA(cfg.model, input_dim=(3, 42, 42), action_dim=1)

from mawm.models.misc import ObsPred, MsgPred
from mawm.models.vision import SemanticEncoder
obs_pred = ObsPred(h_dim=32, out_channels=18)
msg_pred = MsgPred(h_dim=32, in_channels=16)
msg_encoder = SemanticEncoder(latent_dim=32)


import torch
ckpt = torch.load("../nbs/models/best.pth", map_location='cpu')
ckpt.keys()


model.load_state_dict(ckpt['jepa'])
msg_encoder.load_state_dict(ckpt['msg_encoder'])
msg_pred.load_state_dict(ckpt['msg_predictor'])
obs_pred.load_state_dict(ckpt['obs_predictor'])


planner = CEMPlanner(model=model, msg_enc= msg_encoder, msg_pred=msg_pred, obs_pred=obs_pred)

evaluator = PlanEvaluator(model, msg_encoder, msg_pred, planner)
evaluator.eval_all_agents(env, preprocessor, negotiation_rounds=3)