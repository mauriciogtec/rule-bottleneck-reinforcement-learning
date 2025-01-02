from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, SupportsFloat

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
            "When remaining number of alerts given budget is 0, the only valid action is 0."
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

    @property
    def task_text(self) -> str:
        return (
            "You are assisting doctors from a hospital in making optimized"
            "decisions about which patient should receive a vital sign monitor device."
            "You will determine the device allocation by considering the patients' current"
            "vital sign, the patients' vital sign variance in the past five timesteps."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A vector which contains a subset of the indices of patients currently in"
            "the system. Each patient whose index appears in the vector will be"
            "assigned a device."
        )

    def state_descriptor(self, *_, **__) -> str:
        """
        Convert the observation into a text description specific to the environment

        Returns:
            str: The text description of the observation
        """

        agent_descriptions = []
        env = self.env
        agent_states = env.agent_states

        for i in range(len(agent_states)):
            agent_index = i  ## Note this is not agent id, but index in the current list
            agent_state = agent_states[i]["state"]
            pulse_rate_value = agent_state[0]
            respiratory_rate_value = agent_state[1]
            spo2_value = agent_state[2]
            pulse_rate_variance = agent_state[3]
            respiratory_rate_variance = agent_state[4]
            spo2_variance = agent_state[5]
            device_flag = agent_state[6]
            time_since_joined = agent_state[7]

            description = f"""
            Agent {agent_index}:
            - Pulse Rate Value: {pulse_rate_value}
            - Respiratory Rate Value: {respiratory_rate_value}
            - SPO2 Value: {spo2_value}
            - Pulse Rate Variance: {pulse_rate_variance}
            - Respiratory Rate Variance: {respiratory_rate_variance}
            - SPO2 Variance: {spo2_variance}
            - Device Allocation Flag: {device_flag}
            - Time Slot Since Joined: {time_since_joined}
            """
            agent_descriptions.append(description.strip())
        return "\n\n".join(agent_descriptions)

    # def language_descriptor(self, obs: Any, info: Dict[str, Any]) -> str:
    #     """
    #     Convert the observation into a text description specific to the VitalSigns environment.

    #     Args:
    #         obs (Any): The observation to convert into text.
    #         info (dict[str, Any]): Additional information about the observation.

    #     Returns:
    #         str: The text description of the observation.
    #     """
    #     raise NotImplementedError("VitalSignsWrapper is not implemented yet.")


class FinanceWrapper(LanguageWrapper):
    """
    A wrapper for the Finance environment.
    """

    state_template = (
        "## Initial outlook\n"
        "{}\n\n"
        "## Current date\n"
        "{}\n\n"
        "## Last week prices from current date\n"
        "{}\n\n"
        "## Has bought? If so, price and date\n"
        "{}"
    )

    @property
    def task_text(self) -> str:
        return (
            "You are assisting a financial analyst in making optimized decisions about"
            " when two buy or sell a single stock. You will determine the action by"
            " considering the current stock price, the stock price history, the"
            " analyst's predictions, and news articles about the stock."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A single integer value representing the decision:"
            "0 = buy the stock\n"
            "1 = sell the stock\n"
            "2 = hold the stock\n"
        )

    def state_descriptor(self, obs, info):
        """
        Convert the observation into a text description specific to the Finance environment.

        Args:
            obs (Any): The observation to convert into text.
            info (dict[str, Any]): Additional information about the observation.

        Returns:
            str: The text description of the observation.
        """

        initial_outlook = info["initial_outlook"]
        current_date = info["current_date"]
        last_week_prices = info["last_prices"]
        has_bought = info["has_bought"]
        if has_bought:
            buying_price = info["buying_price"]
            buying_date = info["buying_date"]
            msg = f"Yes, bought at {buying_price} on {buying_date}"
        else:
            msg = "No"

        text_state = self.state_template.format(
            initial_outlook, current_date, last_week_prices, msg
        )

        return text_state


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
