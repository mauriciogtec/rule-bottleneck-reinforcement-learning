from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, SupportsFloat

import numpy as np
from gymnasium import Env, Wrapper
from gymnasium.core import ActType
from numpy import ndarray


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
        embeddings_model (TogetherEmbeddings): The language model used to embed the text descriptions.
    """

    def __init__(self, env: Env, embeddings_model: TogetherEmbeddings) -> None:
        super().__init__(env)
        self.embeddings_model = embeddings_model

    @abstractmethod
    def language_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        """
        Convert the observation into a text description.

        Args:
            obs (ndarray): The observation to convert into text.
            info (dict[str, Any]): Additional information about the observation.

        Returns:
            str: The text description of the observation.
        """
        pass

    def step(self, action: ActType) -> Tuple[ndarray, SupportsFloat, bool, bool, Dict[Any, Any]]:
        """
        Take a step in the environment using the given action.

        Args:
            action (ActType): The action to take.

        Returns:
            tuple: A tuple containing the embedded observation, reward, termination status,
                   truncation status, and additional info.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        desc = self.language_descriptor(obs, info)
        info["obs_text"] = desc
        obs = self.embeddings_model.embed_query(desc)
        obs = np.array(obs, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def reset(self) -> Tuple[ndarray, Dict[str, Any]]:
        """
        Reset the environment.

        Returns:
            tuple: A tuple containing the embedded initial observation and additional info.
        """
        obs, info = self.env.reset()
        desc = self.language_descriptor(obs, info)
        info["obs_text"] = desc
        obs = self.embeddings_model.embed_query(desc)
        obs = np.array(obs, dtype=np.float32)
        return obs, info


class HeatAlertsWrapper(LanguageWrapper):
    """
    A wrapper for the HeatAlerts environment from Considine et al. (2024).
    """

    def language_descriptor(self, *_, **__) -> str:
        """
        Convert the observation into a text description specific to the HeatAlerts environment.

        Returns:
            str: The text description of the observation.
        """
        template = (
            "- Location (FIPS code): {} "
            "\n- Remaining number of alerts: {} "
            "\n- Current date (day of summer): {} (day {} of 152) "
            "\n- Current heat index: {}% "
            "\n- Average heat index over the past 3 days: {}% "
            "\n- Excess heat compared to the last 3 days: {}% "
            "\n- Excess heat compared to the last 7 days: {}% "
            "\n- Weekend (yes/no): {} "
            "\n- Holiday (yes/no): {} "
            "\n- Alerts in last 14 days: {} "
            "\n- Alerts in last 7 days: {} "
            "\n- Alerts in last 3 days: {} "
            "\n- Alert streak: {} "
            "\n- Heat index forecast for next 14 days: {} "
            "\n- Max forecasted per week for the rest of the summer: {}"
        )
        env = self.env
        date = env.ep.index[env.t]
        obs = env.observation
        ep = env.ep

        # get the forecasted heat index for the next 14 days as a dict
        f14 = ep["heat_qi"].iloc[env.t + 1 : env.t + 14]
        f14 = ((100 * f14).round(2).astype(int).astype(str) + "%").to_dict()
        f14 = "\n  * ".join([f"{k}: {v}" for k, v in f14.items()])

        # forecast per remaining weeks
        ep_ = ep.iloc[env.t + 1 :].copy()
        ep_["week"] = ep_.index.str.slice(0, 7)
        heat_qi_weekly = ep_.groupby("week")["heat_qi"].max()
        heat_qi_weekly = (
            (100 * heat_qi_weekly).round(2).astype(int).astype(str) + "%"
        ).to_dict()
        heat_qi_weekly = "\n  * ".join([f"{k}: {v}" for k, v in heat_qi_weekly.items()])

        return template.format(
            env.location,
            obs.remaining_budget,
            date,
            env.t,
            int(100 * obs.heat_qi.round(2)),
            int(100 * obs.heat_qi_3d.round(2)),
            int(100 * obs.excess_heat_3d.round(2)),
            int(100 * obs.excess_heat_7d.round(2)),
            "yes" if obs.weekend else "no",
            "yes" if obs.holiday else "no",
            sum(env.actual_alert_buffer[-14:]) if env.t > 1 else 0,
            sum(env.actual_alert_buffer[-7:]) if env.t > 1 else 0,
            sum(env.actual_alert_buffer[-3:]) if env.t > 1 else 0,
            env.alert_streak,
            f14,
            heat_qi_weekly,
        )


class VitalSignsWrapper(LanguageWrapper):
    """
    A wrapper for the VitalSigns environment.
    """

    def language_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
        """
        Convert the observation into a text description specific to the VitalSigns environment.

        Args:
            obs (Any): The observation to convert into text.
            info (dict[str, Any]): Additional information about the observation.

        Returns:
            str: The text description of the observation.
        """
        raise NotImplementedError("VitalSignsWrapper is not implemented yet.")


if __name__ == "__main__":
    # Example usage
    from weather2alert.env import HeatAlertEnv
    from langchain_together import TogetherEmbeddings

    model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    env = HeatAlertEnv()

    wrapped_env = HeatAlertsWrapper(env, model)

    obs, info = wrapped_env.reset()
    print(info["obs_text"])

    for _ in range(10):
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(info["obs_text"])
