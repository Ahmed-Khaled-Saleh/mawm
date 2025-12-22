import numpy as np
import gym
from mawm.envs.marl_grid._envs import env_from_config


class QLearningAgent:
    def __init__(
        self,
        action_space_n,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay_rate=0.995,
    ):
        # def __init__(self, action_space_n, learning_rate=0.01, discount_factor=0.95, exploration_rate=0.5, exploration_decay_rate=0.995):
        self.action_space_n = action_space_n
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.initial_exploration_rate = exploration_rate
        self.min_exploration_rate = 0.01  # Minimum exploration rate
        self.q_table = {}

    def update_exploration_rate(self):
        # Decay the exploration rate over time but maintain a minimum exploration rate
        self.exploration_rate *= self.exploration_decay_rate
        self.exploration_rate = max(self.exploration_rate, self.min_exploration_rate)

    def state_to_key(self, state):
        # Convert the state to a string (or another hashable type) to use as a dictionary key
        return str(state)

    def select_action(self, state):
        state_key = self.state_to_key(state)
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(self.action_space_n)
        else:
            return self.get_best_action(state_key)

    def get_best_action(self, state_key):
        # Get the best action for a given state from the Q-table
        state_actions = self.q_table.get(state_key, {})
        if state_actions:
            return max(state_actions, key=state_actions.get)
        else:
            # If the state is not in the Q-table, choose a random action
            return np.random.choice(self.action_space_n)

    def update_q_table(self, state, action, reward, next_state):
        state_key = self.state_to_key(state)
        next_state_key = self.state_to_key(next_state)

        # Initialize the Q-values for the state if it's not already in the Q-table
        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in range(self.action_space_n)}

        # Calculate the maximum future reward
        max_future_reward = max(
            self.q_table.get(next_state_key, {}).values(), default=0
        )

        # Update the Q-value for the state-action pair
        current_q_value = self.q_table[state_key][action]
        new_q_value = current_q_value + self.learning_rate * (
            reward + self.discount_factor * max_future_reward - current_q_value
        )
        self.q_table[state_key][action] = new_q_value

    def update_exploration_rate(self):
        # Decay the exploration rate over time
        self.exploration_rate *= self.exploration_decay_rate


# # Example usage
# env_config = {
#     "env_class": "ClutteredGoalCycleEnv",
#     "grid_size": 7,
#     "max_steps": 80000,
#     "clutter_density": 0.10,
#     "respawn": True,
#     "ghost_mode": True,
#     "reward_decay": False,
#     "n_bonus_tiles": 3,
#     "initial_reward": True,
#     "penalty": -1.5,
#     "agents": [
#         {
#             "view_size": 7,
#             "view_offset": 1,
#             "view_tile_size": 11,
#             "observation_style": "rich",
#             "color": "red",
#         },
#         {
#             "view_size": 7,
#             "view_offset": 1,
#             "view_tile_size": 11,
#             "observation_style": "rich",
#             "color": "blue",
#         }
#         # Add more agent configurations if needed
#     ],
# }
# # Initialize the environment
# env = env_from_config(env_config)


from mawm.envs.marl_grid.envs.findgoal_env.environments import make_environment
from mawm.envs.marl_grid.envs.findgoal_env.cfgs import config
config.env_cfg.seed = 42
config.env_cfg.max_steps = 10000
create_env = lambda: make_environment(config.env_cfg)

env = create_env()


# agents = [f'agent_{i}' for i in range(config.env_cfg.num_agents)]


# Create a QLearningAgent for each agent in the environment
import pdb; pdb.set_trace()
agents = [
    QLearningAgent(action_space_n=env.action_space.spaces[i].n)
    for i in range(len(env.agents))
]

# Start an episode
states = env.reset()
done = False

while not done:
    # Reset exploration rate at the beginning of each episode
    for agent in agents:
        agent.exploration_rate = agent.initial_exploration_rate

    env.render()

    # Get actions for all agents
    actions = [agents[i].select_action(states[i]) for i in range(len(env.agents))]

    # Environment steps
    next_states, rewards, done, _ = env.step(actions)

    # Update Q-table for each agent
    for i in range(len(env.agents)):
        agents[i].update_q_table(states[i], actions[i], rewards[i], next_states[i])
        agents[i].update_exploration_rate()

    # Update states
    states = next_states

# Episode finished
print("Episode finished")