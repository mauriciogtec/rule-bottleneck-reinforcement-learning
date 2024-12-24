from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, SupportsFloat

import numpy as np
from gymnasium import Env, Wrapper
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

    def __init__(self, env: Env, embeddings_model: Embeddings) -> None:
        super().__init__(env)
        self.embeddings_model = embeddings_model

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
        desc = self.state_descriptor(obs, info)
        info["obs_text"] = desc
        obs = self.embeddings_model.embed_query(desc)
        obs = np.array(obs, dtype=np.float32)
        return obs, info


class HeatAlertsWrapper(LanguageWrapper):
    """
    A wrapper for the HeatAlerts environment from Considine et al. (2024).
    """

    @property
    def task_text(self) -> str:
        return (
            "You are assisting officials from the National Weather Service in making optimized"
            " decisions about when to issue public heatwave alerts. You will determine whether"
            " to issue an alert by considering multiple factors related to current weather conditions,"
            " past alert history, and the remaining number of alerts for the season."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A single integer value representing the decision:"
            "1 = issue an alert"
            "0 = do not issue an alert"
        )

    def state_descriptor(self, *_, **__) -> str:
        """
        Convert the observation into a text description specific to the HeatAlerts environment.

        Returns:
            str: The text description of the observation.
        """
        template = (
            "- Location (FIPS code): {} "
            "\n- Remaining number of alerts given budget: {} "
            "\n- Current date and day of summer, day of 152): {}, {}"
            "\n- Current heat index (0 to 100%): {}%"
            "\n- Average heat index over the past 3 days (0 to 100%): {}% "
            "\n- Excess heat compared to the last 3 days (0 to 100%): {}% "
            "\n- Excess heat compared to the last 7 days (0 to 100%): {}% "
            "\n- Weekend (yes/no): {} "
            "\n- Holiday (yes/no): {} "
            "\n- Alerts in last 14 days: {} "
            "\n- Alerts in last 7 days: {} "
            "\n- Alerts in last 3 days: {} "
            "\n- Alert streak: {} "
            "\n- Heat index forecast for next 14 days (0 to 100%): {} "
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

    def state_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
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
