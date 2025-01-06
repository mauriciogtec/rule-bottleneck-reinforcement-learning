import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal  # , Optional, Tuple, SupportsFloat

# import numpy as np
from gymnasium import Env, Wrapper, spaces

# from gymnasium.core import ActType
# from numpy import ndarray
# from langchain_core.embeddings import Embeddings


class LanguageWrapper(Wrapper, ABC):
    """
    A wrapper for a gym environment that embeds the observation text using a language model.

    This wrapper takes a gym environment and a language model for embedding text. It processes
    the observations from the environment by converting them into text descriptions and then
    embedding these descriptions using the provided language model. The embedded observations
    are then returned along with the original reward, termination status, truncation status,
    and additional info.

    Args:
        env (gym.Env): The gym environment to wrap.
        embeddings_model (Embeddings): The language model used to embed the text descriptions.
    """

    def __init__(
        self,
        env: Env,
        obs_type: Literal["text", "original", "both"] = "both",
        max_text_length: int = 2048,
        parse_action: bool = False,
        # embeddings_model: Optional[Embeddings] = None,
        # embeddings_dim: int = 768,
    ) -> None:
        super().__init__(env)
        # self.embeddings_model = embeddings_model

        # if self.embeddings_model is not None:
        #     # update obs space
        #     self.env.observation_space = spaces.Box(
        #         low=-np.inf, high=np.inf, shape=(embeddings_dim,)
        #     )
        self.obs_type = obs_type

        if self.obs_type == "text":
            self.env.observation_space = spaces.Text(max_length=max_text_length)
        elif self.obs_type == "both":
            self.env.observation_space = spaces.Tuple(
                (self.env.observation_space, spaces.Text(max_length=2048))
            )

        self.metadata["task_text"] = self.task_text
        self.metadata["action_space_text"] = self.action_space_text
        self.parse_action = parse_action

    @property
    @abstractmethod
    def task_text() -> str:
        """
        Return a description of the task that the environment is solving.

        Returns:
            str: The task description.
        """
        pass

    @property
    @abstractmethod
    def action_space_text() -> str:
        """
        Return a description of the action space of the environment.

        Returns:
            str: The action space description.
        """
        pass

    @abstractmethod
    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        """
        Convert the observation into a text description.

        Args:
            obs (ndarray): The observation to convert into text.
            info (dict[str, Any]): Additional information about the observation.

        Returns:
            str: The text description of the observation.
        """
        pass

    def step(self, action: str):
        """
        Take a step in the environment using the given action.

        Args:
            action (ActType): The action to take.

        Returns:
            tuple: A tuple containing the embedded observation, reward, termination status,
                   truncation status, and additional info.
        """
        # parse action
        if self.parse_action:
            action = self.action_parser(action)

        # take a step in the environment
        obs_original, reward, terminated, truncated, info = self.env.step(action)
        obs_text = self.state_descriptor(obs_original, info)

        if self.obs_type == "text":
            obs = obs_text
        elif self.obs_type == "original":
            obs = obs_original
        elif self.obs_type == "both":
            obs = (obs_original, obs_text)
        else:
            raise ValueError("Invalid observation type")

        obs_text = self.state_descriptor(obs_original, info)

        # info["obs_text"] = desc

        # if self.embeddings_model is not None:
        #     obs = self.embeddings_model.embed_query(desc)
        #     obs = np.array(obs, dtype=np.float32)

        # self.env.metadata["last_obs"] = obs
        # self.env.metadata["last_info"] = info
        # self.env.metadata["obs_text"] = desc

        self.env.metadata["obs_original"] = info["obs_original"] = obs_original
        self.env.metadata["obs_text"] = info["obs_text"] = obs_text

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options={}):
        """
        Reset the environment.

        Returns:
            tuple: A tuple containing the embedded initial observation and additional info.
        """
        obs_original, info = self.env.reset(seed=None)
        # TODO: add support for the options parameter

        obs_text = self.state_descriptor(obs_original, info)

        if self.obs_type == "text":
            obs = obs_text
        elif self.obs_type == "original":
            obs = obs_original
        elif self.obs_type == "both":
            obs = (obs_original, obs_text)
        else:
            raise ValueError("Invalid observation type")

        self.env.metadata["obs_original"] = info["obs_original"] = obs_original
        self.env.metadata["obs_text"] = info["obs_text"] = obs_text

        return obs, info

    def action_parser(self, s: str) -> int | List[int]:
        """
        Convert the action into a text description.

        Args:
            s: The action string to convert into text.

        Returns:
            str: The text description of the action.
        """
        # Uses regex to extract all integers
        # If only one is found, returns it as an integer
        #   otherwise, returns a list of integers
        out = [int(i) for i in re.findall(r"\d+", s)]
        return out[0] if len(out) == 1 else out
