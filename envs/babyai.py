from typing import Any, Dict, List
import gymnasium as gym
import numpy as np
from minigrid.core.constants import (
    OBJECT_TO_IDX,
    COLOR_TO_IDX,
    IDX_TO_OBJECT,
)
from minigrid.wrappers import SymbolicObsWrapper
from envs.wrappers import LanguageWrapper


class BaseBabyAILang(LanguageWrapper):
    """
    Base class for BabyAI language wrappers.
    This class wraps the environment with SymbolicObsWrapper and produces structured text observations,
    including agent position, direction, found objects, unseen areas, and known walls.
    """

    DIR_TO_NAME = ["right", "down", "left", "up"]

    def __init__(self, env_name: str, **kwargs):
        env = gym.make(env_name, **kwargs)
        current_obs_space = env.observation_space
        h, w = current_obs_space["image"].shape[0:2]
        env = SymbolicObsWrapper(env)
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Box(
                    low=0, high=255, shape=(h * w + 2,), dtype=np.int32
                ),  # +2 for color and type
                gym.spaces.Text(max_length=2048),
            )
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
            "Respond in JSON format. Example: {'action': 1}"
        )

    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        """
        Generates a text description of the symbolic state observation.
        Includes mission, agent's absolute position and direction, and lists of seen objects, walls, and unseen cells.
        """
        mission = obs.get("mission", "No mission specified.")
        description = f"Mission: {mission}\n\n"

        grid = obs.get("image", None)
        if grid is None:
            return description

        rows, cols, channels = grid.shape
        agent_position = None
        agent_direction = None
        objects_list = []
        unseen_list = []
        walls_list = []

        for i in range(rows):
            for j in range(cols):
                cell = grid[i, j]
                x, y, obj_id = int(cell[0]), int(cell[1]), int(cell[2])

                if obj_id == OBJECT_TO_IDX["agent"]:
                    agent_position = (x, y)
                elif obj_id == OBJECT_TO_IDX["unseen"]:
                    unseen_list.append((x, y))
                elif obj_id == OBJECT_TO_IDX["wall"]:
                    walls_list.append((x, y))
                elif obj_id in {OBJECT_TO_IDX["door"], OBJECT_TO_IDX["empty"], -1}:
                    continue
                else:
                    obj_name = IDX_TO_OBJECT[obj_id]
                    objects_list.append(((x, y), obj_name))

        # Direction
        dir_idx = getattr(self.env.unwrapped, "agent_dir", None)
        dir_text = (
            f", facing {self.DIR_TO_NAME[dir_idx]}"
            if dir_idx is not None
            else " (direction unknown)"
        )

        if agent_position is not None:
            description += f"You are at position:\n{agent_position}{dir_text}\n\n"
        else:
            description += "Agent position not found.\n\n"

        # Found objects
        description += "You have found objects:\n"
        if objects_list:
            for pos, name in objects_list:
                description += f"  At {pos}: {name}\n"
        else:
            description += "  None\n"

        # Unseen areas
        description += "\nUnseen areas:\n"
        if unseen_list:
            description += ", ".join([f"{pos}" for pos in unseen_list]) + "\n"
        else:
            description += "  None\n"

        # Known walls
        description += "\nKnown walls:\n"
        if walls_list:
            description += ", ".join([f"{pos}" for pos in walls_list])
        else:
            description += "  None\n"

        return description

    def reset(self, seed=None, options: Dict[str, Any] = None):
        (obs_original, obs_text), info = super().reset(seed=seed, options=options)
        numeric = self.numeric_state(obs_original)

        info["original_obs"] = obs_original

        return (numeric, obs_text), info

    def step(self, action: Any):
        if self.parse_action:
            action = self.action_parser(action)

        (obs_original, obs_text), reward, terminated, truncated, info = super().step(action)
        numeric = self.numeric_state(obs_original)
        info["original_obs"] = obs_original
        return (numeric, obs_text), reward, terminated, truncated, info

    def numeric_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Returns a fully numeric state representation.
        Format: [mission_color_idx, mission_type_idx, flattened grid object IDs]
        """
        grid = obs.get("image", None)
        if grid is None:
            return np.array([])

        rows, cols, _ = grid.shape
        object_grid = np.array(
            [[cell[2] for cell in row] for row in grid], dtype=np.int32
        )
        flat_grid = object_grid.flatten()

        # Encode mission like: "go to the red ball"
        mission = obs.get("mission", "").strip().lower()
        tokens = mission.split()
        color_idx = -1
        type_idx = -1

        if len(tokens) >= 4:
            color_str = tokens[-2]
            type_str = tokens[-1]
            color_idx = COLOR_TO_IDX.get(color_str, -1)
            type_idx = OBJECT_TO_IDX.get(type_str, -1)

        state_vector = np.concatenate([[color_idx, type_idx], flat_grid])
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
    print("\nMetadata:")
    print(env.metadata)

    # Test step
    sample_action = "2"  # Move forward
    obs, reward, terminated, truncated, info = env.step(sample_action)
    print("\nAfter action '2':")
    print(obs)
    print("\nReward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("\nMetadata:")
    print(env.metadata)

    env.close()
