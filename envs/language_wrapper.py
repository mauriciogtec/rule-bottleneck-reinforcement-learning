from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
from gymnasium import Env, Wrapper, spaces
from gymnasium.core import ActType
from numpy import ndarray
from langchain_core.embeddings import Embeddings


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
        embeddings_model: Optional[Embeddings] = None,
        embeddings_dim: int = 768,
    ) -> None:
        super().__init__(env)
        self.embeddings_model = embeddings_model

        if self.embeddings_model is not None:
            # update obs space
            self.env.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(embeddings_dim,)
            )

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

    def step(
        self, action: ActType
    ) -> Tuple[ndarray, SupportsFloat, bool, bool, Dict[Any, Any]]:
        """
        Take a step in the environment using the given action.

        Args:
            action (ActType): The action to take.

        Returns:
            tuple: A tuple containing the embedded observation, reward, termination status,
                   truncation status, and additional info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        desc = self.state_descriptor(obs, info)
        info["obs_text"] = desc

        if self.embeddings_model is not None:
            obs = self.embeddings_model.embed_query(desc)
            obs = np.array(obs, dtype=np.float32)

        self.last_obs = obs
        self.last_info = info

        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs) -> Tuple[ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            tuple: A tuple containing the embedded initial observation and additional info.
        """
        obs, info = self.env.reset(*args, **kwargs)
        desc = self.state_descriptor(obs, info)
        info["obs_text"] = desc

        self.last_obs = obs
        self.last_info = info

        if self.embeddings_model is not None:
            obs = self.embeddings_model.embed_query(desc)
            obs = np.array(obs, dtype=np.float32)

        return obs, info
