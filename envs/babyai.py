from functools import partial
from typing import Any, Dict, List
import gymnasium as gym
import numpy as np
from minigrid.core import constants
from minigrid.wrappers import SymbolicObsWrapper, FullyObsWrapper
from envs.wrappers import LanguageWrapper


class BabyAI(gym.Env):
    """
    BabyAI environment wrapper for flattened observations from symbolic observations.
    """

    def __init__(self, env_name: str, **kwargs):
        super().__init__()
        self.env = FullyObsWrapper(gym.make(env_name, **kwargs))
        # self.env = SymbolicObsWrapper(env)
        h, w = self.env.observation_space["image"].shape[0:2]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(h * w * 3 + 3,), dtype=np.float32
        )
        self.action_space = self.env.action_space

    def numeric_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Returns a fully numeric state representation.
        Scans the fully observable. Each grid entry has elements OBJECT_IDX, COLOR_IDX, STATE)
        """
        flat_grid = obs["image"].flatten()

        direction = obs.get("direction", 0)
        # Encode mission like: "go to the red ball"
        mission = obs.get("mission", "").strip().lower()
        tokens = mission.split()
        color_idx = -1
        type_idx = -1

        if len(tokens) >= 4:
            color_str = tokens[-2]
            type_str = tokens[-1]
            color_idx = constants.COLOR_TO_IDX.get(color_str, -1)
            type_idx = constants.OBJECT_TO_IDX.get(type_str, -1)

        state_vector = np.concatenate([[color_idx, type_idx, direction], flat_grid])
        return state_vector

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        numeric = self.numeric_state(obs)
        info["original_obs"] = obs
        return numeric, reward, terminated, truncated, info

    def reset(self, seed=None, options: Dict[str, Any] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        numeric = self.numeric_state(obs)
        info["original_obs"] = obs
        return numeric, info


BabyAIGoToObjNumeric = partial(BabyAI, env_name="BabyAI-GoToObjS6-v1")
BabyAIGoToLocalNumeric = partial(BabyAI, env_name="BabyAI-GoToLocalS8N7-v0")


class BaseBabyAILang(LanguageWrapper):
    """
    Base class for BabyAI language wrappers.
    This class wraps the environment with SymbolicObsWrapper and produces structured text observations,
    including agent position, direction, found objects, unseen areas, and known walls.
    """

    DIR_TO_NAME = ["right", "down", "left", "up"]
    IDX_TO_STATE = dict(
        zip(constants.STATE_TO_IDX.values(), constants.STATE_TO_IDX.keys())
    )

    def __init__(self, env_name: str, **kwargs):
        env = gym.make(env_name, **kwargs)
        # current_obs_space = env.observation_space
        # env = SymbolicObsWrapper(env)
        # h, w = current_obs_space["image"].shape[0:2]
        # env.observation_space = gym.spaces.Box(
        #     low=-np.inf, high=np.inf, shape=(h * w + 3,), dtype=np.float32
        # )  # +2 for color and type
        env = FullyObsWrapper(env)
        h, w = env.observation_space["image"].shape[0:2]
        self.h, self.w = h, w
        env.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(h * w * 3 + 3,), dtype=np.float32
        )
        super().__init__(env)

    @property
    def action_space_text(self) -> str:
        return (
            "Discrete actions available:\n"
            "0: Turn left\n"
            "1: Turn right\n"
            "2: Move forward\n"
            "3: Pick up object\n"
            "4: Drop object\n"
            "5: Toggle/open door (if applicable)\n\n"
            "\nExample of action effects:"
            "- If you are at (2, 3) and you are facing right, then moving action=2 (forward) will take you to (2, 4).\n"
            "- If you are at (2, 3) and you are facing down, then moving action=2 (forward) will take you to (3, 3).\n"
            "- If you are at (2, 3) and there is a key at (2, 4) and facing right towards the key, then you can pickup the key with action=3 (pick up object).\n"
            "- If you are at (2, 3) facing down and you want to move to (2, 4), then you first need to turn left (action=1) and then move forward (action=2).\n"
            "- If you are at (2, 3) facing left and you want to move to (3, 3), then you first need to turn left (action=0) and then move forward (action=2).\n"
            "- If you are at (2, 3) facing down and you want to move to (2, 2), then you first need to turn right (action=1) and then move forward (action=2).\n\n"
            "- If you are at (2, 3) facing right and there is a red ball at (4, 3), then you will win by moving left (action=0) and then forward (action=2) to reach (3, 3) since you will be now facing the red ball.\n"
            "You cannot go beyond the limits of grid size from 1 to size - 2. For instance, (2, 1) -> (2, 0) is invalid. (4, 3) -> (5, 3) is invalid when grid size is 6x6.\n"
            "You can only execute one action at a time."
        )

    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        """
        Generates a text description of the symbolic state observation.
        Includes mission, agent's absolute position and direction, and lists of seen objects, walls, and unseen cells.
        """
        mission = obs.get("mission", "No mission specified.")
        description = f"Mission: {mission}\n"

        grid = obs.get("image", None)
        if grid is None:
            return description

        rows, cols, channels = grid.shape
        agent_position = None
        agent_direction = None
        objects_list = []
        unseen_list = []
        walls_list = []
        door_list = []

        for i in range(rows):
            for j in range(cols):
                obj_id, color_id, state_id = grid[i, j]
                pos = (j, i)

                object = constants.IDX_TO_OBJECT[obj_id]

                if object == "agent":
                    agent_position = pos
                    agent_direction = self.DIR_TO_NAME[state_id]
                elif object == "wall":
                    if 0 < i < rows - 1 and 0 < j < cols - 1:
                        walls_list.append(pos)
                elif object == "unseen":
                    unseen_list.append(pos)
                elif object == "door":
                    state = self.IDX_TO_STATE[state_id]
                    door_list.append((pos, state))
                elif object != "empty":
                    color = constants.IDX_TO_COLOR[color_id]
                    objects_list.append((pos, object, color))
                

        description = f"Mission: {mission}\n\n"

        # Agent position
        description += (
            f"You are in a {self.h} x {self.w} grid at position {agent_position} facing {agent_direction}\n"
        )

        # Objects
        if objects_list:
            description += "You see objects:\n"
            for pos, obj, color in objects_list:
                description += f"- {color} {obj} at position {pos}\n"

        # Unseen areas
        if unseen_list:
            description += (
                "Unseen areas:\n" + ", ".join([f"{pos}" for pos in unseen_list]) + "\n"
            )

        if door_list:
            description += "Doors:\n"
            for pos, state in door_list:
                description += f"  Door at {pos}: {state})\n"

        # Known walls
        if walls_list:
            description += "Walls:\n" + ", ".join(
                [f"{pos}" for pos in walls_list]
            )

        # # Direction
        # dir_idx = obs.get("direction", None)
        # dir_text = (
        #     f", facing {self.DIR_TO_NAME[dir_idx]}"
        #     if dir_idx is not None
        #     else " (direction unknown)"
        # )

        # if agent_position is not None:
        #     description += f"You are at position:\n{agent_position}{dir_text}\n\n"
        # else:
        #     description += "Agent position not found.\n\n"

        # # Found objects
        # description += "You have found objects:\n"
        # if objects_list:
        #     for pos, name in objects_list:
        #         description += f"  At {pos}: {name}\n"
        # else:
        #     description += "  None\n"

        # # Unseen areas
        # description += "\nUnseen areas:\n"
        # if unseen_list:
        #     description += ", ".join([f"{pos}" for pos in unseen_list]) + "\n"
        # else:
        #     description += "  None\n"

        # # Known walls
        # description += "\nKnown walls:\n"
        # if walls_list:
        #     description += ", ".join([f"{pos}" for pos in walls_list])
        # else:
        #     description += "  None\n"

        return description

    def reset(self, seed=None, options: Dict[str, Any] = None):
        (obs_original, obs_text), info = super().reset(seed=seed, options=options)
        numeric = self.numeric_state(obs_original)

        info["original_obs"] = obs_original

        return (numeric, obs_text), info

    def step(self, action: Any):
        if self.parse_action:
            action = self.action_parser(action)

        (obs_original, obs_text), reward, terminated, truncated, info = super().step(
            action
        )
        numeric = self.numeric_state(obs_original)
        info["original_obs"] = obs_original
        return (numeric, obs_text), reward, terminated, truncated, info

    def numeric_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Returns a fully numeric state representation.
        Scans the fully observable. Each grid entry has elements OBJECT_IDX, COLOR_IDX, STATE)
        """
        flat_grid = obs["image"].flatten()

        direction = obs.get("direction", 0)
        # Encode mission like: "go to the red ball"
        mission = obs.get("mission", "").strip().lower()
        tokens = mission.split()
        color_idx = -1
        type_idx = -1

        if len(tokens) >= 4:
            color_str = tokens[-2]
            type_str = tokens[-1]
            color_idx = constants.COLOR_TO_IDX.get(color_str, -1)
            type_idx = constants.OBJECT_TO_IDX.get(type_str, -1)

        state_vector = np.concatenate([[color_idx, type_idx, direction], flat_grid])

        return state_vector


class BabyAIGoToObjLang(BaseBabyAILang):
    """
    Language wrapper for the BabyAI GoToObj environment.
    """

    def __init__(self, **kwargs):
        super().__init__("BabyAI-GoToObjS6-v1", **kwargs)

    @property
    def task_text(self) -> str:
        return (
            "Navigate the grid to reach the target object specified by the mission. "
            "The mission is provided in plain language (e.g., 'go to the red ball')."
        )

    @property
    def example_rules(self) -> List[str]:
        return [
            '{"background": "Red ball is in sight. Path looks clear. No wall in front. Close to target.", "rule": "If red ball seen ahead, move forward."}',
            '{"background": "Wall blocks path. Can’t move forward. Right side might be clear.", "rule": "If wall ahead, turn right and go."}',
            '{"background": "Nothing visible ahead. Object is not in view. Grid may hide target.", "rule": "If no target in sight, turn left until seen."}',
        ]


class BabyAIGoToLocalLang(BaseBabyAILang):
    """
    Language wrapper for the BabyAI GoToLocal environment.
    """

    def __init__(self, **kwargs):
        super().__init__("BabyAI-GoToLocalS8N7-v0", **kwargs)

    @property
    def task_text(self) -> str:
        return (
            "Navigate the grid to reach the locally specified target object. "
            "The mission provides a cue based on nearby objects you must approach."
        )

    @property
    def example_rules(self) -> List[str]:
        return [
            '{"background": "Blue key nearby. No block ahead. Target seems close. Move seems safe.", "rule": "If blue key in view, go forward."}',
            '{"background": "Door blocks front. Left is open. Try new path. Avoid blocked way.", "rule": "If door ahead, turn left and move."}',
            '{"background": "No target nearby. Rotate to explore. Room still unexplored.", "rule": "If target unseen, turn until found."}',
        ]


if __name__ == "__main__":
    # Test script
    try:
        env = BabyAIGoToObjLang()
    except Exception as e:
        print("Error creating environment:", e)
        exit(1)

    obs, info = env.reset()
    print("Initial Observation:")
    print(obs[1])
    # print("\nMetadata:")
    # print(env.metadata)


    for i in range(20):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, info = env.step(action)
        # if reward > 0:
        print(f"\nStep {i + 1}:")
        print(f"\nObservation: {obs[1]}")
        print(f"\nReward: {reward}")
        print(f"\nAction: {action}")
        print("\nNext Observation:")
        print(next_obs[1])
        print("Reward:", reward)
        print("Done:",  terminated)
        obs = next_obs
        
        if terminated or truncated:
            obs, info = env.reset()
