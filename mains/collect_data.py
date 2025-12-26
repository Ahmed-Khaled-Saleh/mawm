
import copy
import os
import numpy as np
from mawm.envs.marl_grid import make_env
from mawm.envs.marl_grid.cfg import config

def collect_one_rollout_numpy(args):
    rollout_idx, seed, seed_steps, data_dir = args

    cfg = copy.deepcopy(config)
    cfg.env_cfg.seed = int(seed)
    cfg.env_cfg.max_steps = seed_steps

    env = make_env(cfg.env_cfg)
    agents = [f"agent_{i}" for i in range(cfg.env_cfg.num_agents)]
    obs = env.reset()
    goal_pos = obs["global"]["goal_pos"]
    goal_obs = np.array([
        env.get_goal(env.agents[i], goal_pos)[0]
        for i in range(config.env_cfg.num_agents)
    ])
    
    layout = env.get_layout(render_kwargs={"tile_size":11})

    agent_data = {
        ag: {k: [] for k in ["obs", "act", "next_obs", "selfpos", "orientation", "sees_goal", "rew", "done"]}
        for ag in agents
    }

    episode_len = 0
    success = False
    success_at = -1

    for t in range(seed_steps):
        current_obs = obs
        actions = {ag: env.action_space.sample() for ag in agents}
        obs, rew, done, info = env.step(actions)
        
        for ag in agents:
            agent_data[ag]["obs"].append(current_obs[ag]["pov"])# list of ndarrays
            agent_data[ag]["act"].append(actions[ag]) # actions are list of int64
            agent_data[ag]["next_obs"].append(obs[ag]["pov"])# list of ndarrays

            agent_data[ag]["selfpos"].append(obs[ag]["selfpos"])# list of ndarrays
            agent_data[ag]["orientation"].append(obs[ag]["orientation"])# list of int64
            agent_data[ag]["sees_goal"].append(info[ag]["sees_goal"])# list of np.int64

            agent_data[ag]["rew"].append(rew[ag])# list of float64
            agent_data[ag]["done"].append(info[ag]["done"])# list of bools

        episode_len += 1
        if done["__all__"]:
            success = all(info[ag]["done"] for ag in agents)
            success_at = t if success else -1
            break

    env.close()
    
    save_path = os.path.join(data_dir, f"rollout_{rollout_idx}.npz")
    os.makedirs(data_dir, exist_ok=True)
    save_dict = {}

    for ag in agents:
        save_dict[f"{ag}_obs"] = np.stack(agent_data[ag]["obs"]).astype(np.uint8)
        save_dict[f"{ag}_act"] = np.asarray(agent_data[ag]["act"])
        save_dict[f"{ag}_next_obs"] = np.stack(agent_data[ag]["next_obs"]).astype(np.uint8)
        save_dict[f"{ag}_done"] = np.asarray(agent_data[ag]["done"])

        save_dict[f"{ag}_rew"] = np.asarray(agent_data[ag]["rew"])
        save_dict[f"{ag}_selfpos"] = np.stack(agent_data[ag]["selfpos"])
        save_dict[f"{ag}_orientation"] = np.asarray(agent_data[ag]["orientation"])
        save_dict[f"{ag}_sees_goal"] = np.asarray(agent_data[ag]["sees_goal"])

    save_dict['goal_pos'] = goal_pos
    save_dict['episode_len'] = episode_len
    save_dict['success'] = success
    save_dict['success_at'] = success_at
    save_dict['seed'] = config.env_cfg.seed
    save_dict['layout'] = layout
    save_dict['goal_obs'] = goal_obs

    np.savez_compressed(save_path, **save_dict)
    print(f"> Saved rollout {rollout_idx} to {save_path}")
    
    return rollout_idx
    

from multiprocessing import Pool, cpu_count

def generate_parallel(
    rollouts=100,
    seed_steps=4000,
    data_dir="../datasets/marl_grid_data_v3",
    workers=None,
    num_seeds=5,
):
    os.makedirs(data_dir, exist_ok=True)

    print("Number of workers:", workers)

    seeds = np.tile(np.arange(num_seeds), int(np.ceil(rollouts / num_seeds)))
    seeds = seeds[:rollouts]
    np.random.shuffle(seeds)
    
    args = [
        (i, seeds[i], seed_steps, data_dir)
        for i in range(rollouts)
    ]

    with Pool(processes=workers) as p:
        for idx in p.imap_unordered(collect_one_rollout_numpy, args):
            print(f"✓ rollout {idx} done")

if __name__ == "__main__":
    generate_parallel(
        rollouts=10000,
        seed_steps=2000,
        data_dir="/scratch/project_2009050/datasets/MarlGridV3",
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        num_seeds=5,
    )