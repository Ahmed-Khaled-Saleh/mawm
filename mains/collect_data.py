
import os
import numpy as np
from datasets import Dataset
from mawm.envs.marl_grid import make_env
from mawm.envs.marl_grid.cfg import config
from datasets import Features, Value, Array3D, List
import copy

def collect_one_rollout_ds(args):
    rollout_idx, seed, seed_steps, data_dir = args

    cfg = copy.deepcopy(config)
    cfg.env_cfg.seed = seed
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

    # Initialize data dictionary
    agent_data = {
        ag: {k: [] for k in ["obs", "next_obs", "selfpos", "orientation", "sees_goal", "act", "rew", "done"]}
        for ag in agents
    }

    episode_len = 0
    success = False
    success_at = -1

    prev_obs = obs
    for t in range(seed_steps):
        actions = {ag: env.action_space.sample() for ag in agents}
        obs, rew, done, info = env.step(actions)

        for ag in agents:
            agent_data[ag]["obs"].append(prev_obs[ag]["pov"])
            agent_data[ag]["next_obs"].append(obs[ag]["pov"])
            agent_data[ag]["selfpos"].append(obs[ag]["selfpos"])
            agent_data[ag]["orientation"].append(obs[ag]["orientation"])
            agent_data[ag]["sees_goal"].append(info[ag]["sees_goal"])
            agent_data[ag]["act"].append(actions[ag])
            agent_data[ag]["rew"].append(rew[ag])
            agent_data[ag]["done"].append(info[ag]["done"])

        prev_obs = obs
        episode_len += 1
        if done["__all__"]:
            success = all(info[ag]["done"] for ag in agents)
            success_at = t if success else -1
            break

    env.close()
 
    rows = []

    T = len(agent_data[agents[0]]["obs"])

    for t in range(T):
        row = {}

        for ag in agents:
            row[f"{ag}_obs"] = agent_data[ag]["obs"][t].astype(np.float32)
            row[f"{ag}_next_obs"] = agent_data[ag]["next_obs"][t].astype(np.float32)
            row[f"{ag}_selfpos"] = agent_data[ag]["selfpos"][t].astype(np.float32)
            row[f"{ag}_orientation"] = int(agent_data[ag]["orientation"][t])
            row[f"{ag}_sees_goal"] = bool(agent_data[ag]["sees_goal"][t])
            row[f"{ag}_act"] = int(agent_data[ag]["act"][t])
            row[f"{ag}_rew"] = float(agent_data[ag]["rew"][t])
            row[f"{ag}_done"] = bool(agent_data[ag]["done"][t])

        row['transition_idx'] = int(t)
        row['rollout_idx'] = int(rollout_idx)
        rows.append(row)


    feat_dict = {}

    for ag in agents:
        feat_dict[f"{ag}_obs"] = Array3D(shape=(42, 42, 3), dtype="float32")
        feat_dict[f"{ag}_next_obs"] = Array3D(shape=(42, 42, 3), dtype="float32")
        feat_dict[f"{ag}_selfpos"] = List(Value('float32'))
        feat_dict[f"{ag}_act"] = Value("int32")
        feat_dict[f"{ag}_rew"] = Value("float32")
        feat_dict[f"{ag}_orientation"] = Value("int32")
        feat_dict[f"{ag}_sees_goal"] = Value("bool")
        feat_dict[f"{ag}_done"] = Value("bool")

    feat_dict.update({
        "transition_idx": Value("int32"),
        "rollout_idx": Value("int32"),
    })

    features = Features(feat_dict)    
    dataset = Dataset.from_list(rows, features=features)
    rollout_save_path = os.path.join(data_dir, f"rollout_{rollout_idx}")
    dataset.save_to_disk(rollout_save_path)
    meta_data_path = os.path.join(rollout_save_path, f"metadata_{rollout_idx}.npz")

    meta_data = {}
    meta_data['goal_pos'] = goal_pos
    meta_data['episode_len'] = episode_len
    meta_data['success'] = success
    meta_data['success_at'] = success_at
    meta_data['seed'] = config.env_cfg.seed
    meta_data['layout'] = layout
    meta_data['goal_obs'] = goal_obs

    np.savez(meta_data_path, **meta_data)
    
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
        for idx in p.imap_unordered(collect_one_rollout_ds, args):
            print(f"✓ rollout {idx} done")

if __name__ == "__main__":
    generate_parallel(
        rollouts=10000,
        seed_steps=2000,
        data_dir="/scratch/project_2009050/datasets/MarlGridV3",
        workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 1)),
        num_seeds=5,
    )