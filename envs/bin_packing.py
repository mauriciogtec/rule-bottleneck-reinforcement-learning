# Bin packing environment from
# https://github.com/awslabs/or-rl-benchmarks/blob/master/Bin%20Packing/src/bin_packing_environment.py
# Modified by Mauricio Tec 03/27/25 to work for gymnasium framework and added a Language wrapper
# Original LICENSE
# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.

from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Any, Dict
from envs.wrappers import LanguageWrapper

"""
STATE:
Number of bags at each level
Item size

ACTION:
Choose bag
"""

BIG_NEG_REWARD = -100
BIG_POS_REWARD = 10


class BinPacking(gym.Env):

    def __init__(self, env_config={}):
        config_defaults = {
            "bag_capacity": 9,
            # "item_sizes": [2, 3],
            # "item_probabilities": [0.8, 0.2],
            "item_sizes": [2, 3, 5],
            "item_probabilities": [0.7, 0.2, 0.1],
            # "time_horizon": 1000,
            "time_horizon": 32,
        }

        for key, val in config_defaults.items():
            val = env_config.get(key, val)
            self.__dict__[key] = val
            if key not in env_config:
                env_config[key] = val
        # print("Using bin size: ", self.bag_capacity)

        self.episode_count = 0

        self.observation_space = spaces.Box(
            low=np.array([0] * self.bag_capacity + [0]),
            high=np.array(
                [self.time_horizon] * self.bag_capacity + [max(self.item_sizes)]
            ),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(self.bag_capacity)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.time_remaining = self.time_horizon
        self.item_size = self._get_item()
        self.num_full_bags = 0
        self.num_bins_levels = [0] * self.bag_capacity

        initial_state = self.num_bins_levels + [self.item_size]
        self.total_reward = 0
        self.waste = 0
        self.episode_count += 1
        self.bin_type_distribution_map = {}
        self.step_count = 0

        initial_state = np.array(initial_state, dtype=np.float32)

        return initial_state, {}

    def step(self, action):
        done = False
        truncated = False
        self.step_count += 1
        if action >= self.bag_capacity:
            # print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            reward = BIG_NEG_REWARD - self.waste
            done = True
        elif action == 0:
            self.num_bins_levels[self.item_size] += 1
            self.waste = self.bag_capacity - self.item_size
            reward = -1 * self.waste
            self._update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # print("cannot insert item because bin of this level does not exist")
            reward = BIG_NEG_REWARD - self.waste
            done = True
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.waste = -self.item_size
            reward = -1 * self.waste
            self._update_bin_type_distribution_map(action)
            if self.num_bins_levels[action] < 0:
                # print(self.num_bins_levels[action])
                pass
            self.num_bins_levels[action] -= 1

        self.total_reward += reward
        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        self.item_size = self._get_item()
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map

        state = np.array(state, dtype=np.float32)

        return state, reward, done, truncated, info

    def _get_item(self):
        num_items = len(self.item_sizes)
        item_index = np.random.choice(num_items, p=self.item_probabilities)
        return self.item_sizes[item_index]

    def _update_bin_type_distribution_map(self, target_bin_util):
        if target_bin_util < 0 or target_bin_util + self.item_size > self.bag_capacity:
            # print("Error: Invalid Bin Utilization/Item Size")
            return
        elif (
            target_bin_util > 0
            and target_bin_util not in self.bin_type_distribution_map
        ):
            # print("Error: bin_type_distribution_map missing key:", target_bin_util)
            return
        elif (
            target_bin_util > 0
            and target_bin_util in self.bin_type_distribution_map
            and len(self.bin_type_distribution_map[target_bin_util]) == 0
        ):
            # print("Error: bin_type_distribution_map has no element at level", target_bin_util)
            return
        elif target_bin_util == 0:
            if self.item_size not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[self.item_size] = {
                    str(self.item_size): 1
                }
            elif (
                str(self.item_size)
                not in self.bin_type_distribution_map[self.item_size]
            ):
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] = 1
            else:
                self.bin_type_distribution_map[self.item_size][str(self.item_size)] += 1
        else:
            key = np.random.choice(
                list(self.bin_type_distribution_map[target_bin_util].keys())
            )
            if self.bin_type_distribution_map[target_bin_util][key] <= 0:
                # print("Error: Invalid bin count!")
                return
            elif self.bin_type_distribution_map[target_bin_util][key] == 1:
                del self.bin_type_distribution_map[target_bin_util][key]
            else:
                self.bin_type_distribution_map[target_bin_util][key] -= 1

            new_key = self.__update_key_for_bin_type_distribution_map(
                key, self.item_size
            )
            if (target_bin_util + self.item_size) not in self.bin_type_distribution_map:
                self.bin_type_distribution_map[target_bin_util + self.item_size] = {
                    new_key: 1
                }
            elif (
                new_key
                not in self.bin_type_distribution_map[target_bin_util + self.item_size]
            ):
                self.bin_type_distribution_map[target_bin_util + self.item_size][
                    new_key
                ] = 1
            else:
                self.bin_type_distribution_map[target_bin_util + self.item_size][
                    new_key
                ] += 1

    @staticmethod
    def __update_key_for_bin_type_distribution_map(key, item_size):
        parts = key.split(" ")
        parts.append(str(item_size))
        parts.sort()
        return " ".join(parts)

    def render(self, mode="human", close=False):
        pass


class BinPackingIncremental(BinPacking):
    def step(self, action):
        done = False
        truncated = False
        if action >= self.bag_capacity:
            # print("Error: Invalid Action")
            raise
        elif action > (self.bag_capacity - self.item_size):
            reward = BIG_NEG_REWARD - self.waste
        elif action == 0:
            self.num_bins_levels[self.item_size] += 1
            self.waste = self.bag_capacity - self.item_size
            reward = -1 * self.waste
            self._update_bin_type_distribution_map(0)
        elif self.num_bins_levels[action] == 0:
            # print("cannot insert item because bin of this level does not exist")
            reward = BIG_NEG_REWARD - self.waste
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self._update_bin_type_distribution_map(action)
            self.num_bins_levels[action] -= 1
            self.waste = -self.item_size
            reward = -1 * self.waste

        self.total_reward += reward
        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        self.item_size = self._get_item()
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        state = np.array(state, dtype=np.float32)
        return state, reward, done, truncated, info


class BinPackingNearActionGymEnvironment(BinPacking):
    def step(self, action):
        done = False
        truncated = False
        invalid_action = not (self.__is_action_valid(action))
        if invalid_action:
            action = self.__get_nearest_valid_action(action)

        reward = self.__insert_item(action)

        self.total_reward += reward
        self.time_remaining -= 1
        if self.time_remaining == 0:
            done = True

        self.item_size = self._BinPackingGymEnvironment__get_item()
        state = self.num_bins_levels + [self.item_size]
        info = self.bin_type_distribution_map
        state = np.array(state, dtype=np.float32)
        return state, reward, done, truncated, info

    def __insert_item(self, action):
        if action == 0:
            self.num_bins_levels[self.item_size] += 1
            self.waste = self.bag_capacity - self.item_size
        else:
            if action + self.item_size == self.bag_capacity:
                self.num_full_bags += 1
            else:
                self.num_bins_levels[action + self.item_size] += 1
            self.num_bins_levels[action] -= 1
            self.waste = -self.item_size
        reward = -1 * self.waste
        self._BinPackingGymEnvironment__update_bin_type_distribution_map(action)
        return reward

    def __get_nearest_valid_action(self, action):
        num_actions = self.bag_capacity
        valid_actions = [
            x
            for x in range(1, num_actions)
            if self.num_bins_levels[x] > 0 and x <= (self.bag_capacity - self.item_size)
        ]
        return min(valid_actions, key=lambda x: abs(x - action)) if valid_actions else 0

    def __is_action_valid(self, action):
        if action >= self.bag_capacity:
            # print("Error: Invalid Action ", action)
            raise
        elif action > (self.bag_capacity - self.item_size):
            # print("cannot insert item because bin overflow")
            return False
        elif action == 0:
            return True
        elif self.num_bins_levels[action] == 0:
            # print("cannot insert item because bin of this level does not exist")
            return False
        else:
            return True


class BinPackingContinuousActionEnv(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([1]), dtype=np.float32
        )

    def step(self, action):
        action = np.clip(action, 0, 1)
        action = int(action * (self.bag_capacity - 1))
        return super().step(action)


class BinPackingActionMaskGymEnvironment(BinPackingNearActionGymEnvironment):
    def __init__(self, env_config={}):
        super().__init__(env_config)
        self.observation_space = spaces.Dict(
            {
                "action_mask": spaces.Box(
                    0, 1, shape=(self.action_space.n,), dtype=np.float32
                ),
                "real_obs": self.observation_space,
            }
        )

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        state = super().reset(seed=seed, options=options)
        valid_actions = self.__get_valid_actions()
        self.action_mask = [
            1 if x in valid_actions else 0 for x in range(self.action_space.n)
        ]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, {}

    def step(self, action):
        state, rew, done, truncated, info = super().step(action)
        valid_actions = self.__get_valid_actions()
        self.action_mask = [
            1 if x in valid_actions else 0 for x in range(self.action_space.n)
        ]
        obs = {
            "action_mask": np.array(self.action_mask),
            "real_obs": np.array(state),
        }
        return obs, rew, done, truncated, info

    def __get_valid_actions(self):
        valid_actions = [
            x
            for x in range(1, self.action_space.n)
            if self.num_bins_levels[x] > 0 and x <= (self.bag_capacity - self.item_size)
        ]
        valid_actions.append(0)
        return valid_actions


class BinPackingLang(LanguageWrapper):
    def __init__(self, **kwargs):
        env = BinPacking(**kwargs)
        super().__init__(env)

    @property
    def task_text(self) -> str:
        return (
            "You are tasked with solving the online bin packing problem. Items of varying sizes arrive one at a time, "
            "and your goal is to place each item into a bin such that the total number of bins used is minimized. "
            f"Each bin has a fixed capacity of {self.env.bag_capacity} units. You must decide whether to place the current item into an "
            "existing bin (based on its fill level) or to start a new bin.\n\n"
            "The reward at each step is the negative of the waste (unused space) created by your decision. "
            "Trying to place an item into an invalid bin (overflow or nonexistent level) results in a large penalty "
            "(-100) and ends the episode. The environment ends after a fixed number of steps.\n\n"
            "Key challenges:\n"
            "- Items arrive sequentially and randomly, and you do not know what items are coming next.\n"
            "- Efficient packing requires anticipating future items and preserving flexibility in bin levels.\n"
            "- Opening too many bins increases total waste, but being too conservative risks overflow.\n"
        )

    @property
    def action_space_text(self) -> str:
        return (
            "You must return an integer between 0 and {self.env.bag_capacity - 1} representing the level:\n"
            "- 'target_level' = 0 means open a new bin and place the item there.\n"
            f"- 'target_level' in 1 to {self.env.bag_capacity - 1} means place the item into a bin currently filled with 'target_level' many units.\n"
            "Only place items where they will fit. For example, placing a size-3 item into a level-8 bin will overflow.\n"
            'Return your action as a JSON dict: {"target_level": <int>}.'
        )

    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        bin_counts = obs[:-1]
        current_item = obs[-1]

        active_bins = [
            f"{int(count)} bin(s) at fill level {level} (space left: {self.env.bag_capacity - level})"
            for level, count in enumerate(bin_counts)
            if count > 0
        ]
        bins_desc = "\n".join(active_bins) if active_bins else "No active bins yet."

        return (
            f"Current item size: {current_item} unit(s).\n"
            f"Bin capacity: {self.env.bag_capacity} units.\n"
            f"Current bins:\n{bins_desc}\n\n"
            "Decide which bin to place the item in. Opening a new bin adds waste equal to the remaining space. "
            "Placing an item into an existing bin reduces waste only if it fills the bin or gets it closer to full. "
            "Invalid placements (overflow or empty bin level) end the episode."
        )

    @property
    def example_rules(self) -> list[str]:
        example1 = (
            '{"background": "Opening a new bin incurs more waste, especially for small items.", '
            '"rule": "If the current item is small and there are partially filled bins where it fits, prefer placing it in an existing bin."}'
            # '"state relevance": "Current item size is 2. There are 3 bins at level 7 with 2 units of space remaining. Opening a new bin would waste 7 units."}'
        )

        example2 = (
            '{"background": "Larger items are harder to fit later on, so bins with exact fit should be prioritized.", '
            '"rule": "If a bin can exactly fit the current item, choose that bin over others to minimize fragmentation."}'
            # '"state relevance": "Current item size is 3. There is one bin at level 6 (3 units left), which fits perfectly. Other bins have more space but would leave 1+ units unused."}'
        )

        example3 = (
            '{"background": "It is sometimes better to open a new bin if existing bins do not have enough room or would create too much waste.", '
            '"rule": "If placing an item in an existing bin would leave too much unused space or would overflow, start a new bin."}'
            # '"state relevance": "Current item size is 4. All existing bins have 2 or fewer units of space. Attempting to fit it would fail or result in -100 penalty."}'
        )

        return [example1, example2, example3]


class BinPackingIncrementalLang(LanguageWrapper):
    def __init__(self, **kwargs):
        env = BinPackingIncremental(**kwargs)
        super().__init__(env)

    @property
    def task_text(self) -> str:
        freqs = {f"size {k}": f"{int(100 * v):0d}%" for k, v in zip(self.env.item_sizes, self.env.item_probabilities)}
        return (
            "You are tasked with solving the online bin packing problem. Items of varying sizes arrive one at a time, "
            "and your goal is to place each item into a bin such that the total number of bins used is minimized. "
            f"Each bin has a fixed capacity of {self.env.bag_capacity} units. You must decide whether to place the current item into an "
            "existing bin (based on its fill level) or to start a new bin.\n\n"
            "The reward at each step is the negative of the incremental waste created by your decision. If an item is placed into an existing bin, "
            "the incremental waste decreases by the size of the item. If it is placed into a new bin, the waste increases by the unused space left in that new bin. "
            f"The expected item sizes and their frequencies are: {freqs}.\n\n"
        )

    @property
    def action_space_text(self) -> str:
        return (
            f"You must return an integer between 0 and {self.env.bag_capacity - 1} representing the level:\n"
            "- 'target_level' = 0 means open a new bin and place the item there.\n"
            f"- 'target_level' in 1 to {self.env.bag_capacity - 1} means place the item into a bin currently filled with 'target_level' many units.\n"
            "Only place items where they will fit. For example, placing a size-3 item into a level-8 bin will overflow.\n"
            'Return your action as a JSON dict: {"target_level": <int>}.'
        )

    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        bin_counts = obs[:-1]
        current_item = obs[-1]

        active_bins = [
            f"{int(count)} bin(s) at fill level {level} (space left: {self.env.bag_capacity - level})"
            for level, count in enumerate(bin_counts)
            if count > 0
        ]
        bins_desc = "\n".join(active_bins) if active_bins else "No active bins yet."

        return (
            f"Current item size: {current_item} unit(s).\n"
            f"Bin capacity: {self.env.bag_capacity} units.\n"
            f"Current bins:\n{bins_desc}\n\n"
            "Decide which bin to place the item in. Opening a new bin adds waste equal to the remaining space. "
            "Placing an item into an existing bin reduces waste only if it fills the bin or gets it closer to full. "
            "Invalid placements (overflow or empty bin level) end the episode."
        )

    @property
    def example_rules(self) -> list[str]:
        example1 = (
            '{"background": "Opening a new bin is never better if you can fit in a current bin.", '
            '"rule": "Pick the first target level such that `target_level + item_size` is minimized and it does not exceed the bin capacity."}'
            # '"state relevance": "Current item size is 2. There are 3 bins at level 7 with 2 units of space remaining. Opening a new bin would waste 7 units."}'
        )

        example2 = (
            '{"background": "We want to minimize wasted space. Thefore choose the target level that would reduce wasted space.", '
            '"rule": "Pick the target level such that `bin_capacity - (target_level + item_size)` is minimized but nonegative."}'
            # '"state relevance": "Current item size is 3. There is one bin at level 6 (3 units left), which fits perfectly. Other bins have more space but would leave 1+ units unused."}'
        )

        example3 = (
            '{"background": "It is important to consider the expected probabilities of the upcoming object sizes.", '
            '"rule": "If an item size is large and it is frequent to observe small sizes, then always open a new bin. This will allow you to fit smaller items later on."}'
            # '"state relevance": "Current item size is 4. All existing bins have 2 or fewer units of space. Attempting to fit it would fail or result in -100 penalty."}'
        )

        return [example1, example2, example3]


# Example usage
if __name__ == "__main__":
    env_config = {
        "bag_capacity": 10,
        "item_sizes": [1, 2, 3, 4],
        "item_probabilities": [0.25, 0.25, 0.25, 0.25],
        "time_horizon": 100,
    }

    # Initialize the original bin packing environment
    env = BinPacking(env_config)

    # Wrap the environment with the language wrapper
    env = BinPackingLang(env)

    # Reset the environment to start a new episode
    obs, info = env.reset()

    print("Initial obs: ", obs[0])

    done = False
    total_reward = 0

    while not done:
        # Sample a random action
        action = env.action_space.sample()

        # Take a step in the environment
        obs, reward, done, truncated, info = env.step(action)

        # Display the state description after taking the action
        print("Action: ", action)
        print("Reward: ", reward)
        print("Done: ", done)
        print("Text Description:\n", info["obs_text"])

        total_reward += reward

    print(f"Total Reward: {total_reward}")


if __name__ == "__main__":


    # Use base environment
    base_env = BinPacking(env_config)

    # Wrap it in the language wrapper
    lang_env = BinPackingLang(base_env, obs_type="both")

    # Reset the environment
    obs, info = lang_env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = lang_env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = lang_env.step(action)
        total_reward += reward
        step_count += 1

        print(
            f"Step {step_count} | Action: {action} | Reward: {reward:.1f} | Done: {terminated or truncated}"
        )
        print("Text Description:\n", info["obs_text"])
        print("-" * 50)

        done = terminated or truncated

    print("Total Reward: ", total_reward)
