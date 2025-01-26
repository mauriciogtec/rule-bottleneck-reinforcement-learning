from typing import List, Optional

import numpy as np

from envs.wrappers import LanguageWrapper


class HeatAlertsLang(LanguageWrapper):
    """
    A wrapper for the HeatAlerts environment from Considine et al. (2024).
    """

    def __init__(
        self,
        **kwargs,
    ):
        try:
            from weather2alert.env import HeatAlertEnv
        except ImportError:
            raise ImportError(
                "The HeatAlertsWrapper requires the weather2alert package to be installed."
                " Please install it using `pip install git+https://github.com/NSAPH-Projects/weather2alert@dev`."
            )
        env = HeatAlertEnv(**kwargs)
        super().__init__(env)

    @property
    def task_text(self) -> str:
        return (
            "You are assisting officials from the National Weather Service in making optimized"
            " decisions about when to issue public heatwave alerts. You will determine whether"
            " to issue an alert by considering multiple factors related to current weather conditions,"
            " past alert history, and the remaining number of alerts for the season.\n\n"
            "The goal is to minimize the cumulative number of hospitalizations due to heat-related illnesses."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A single integer value representing the decision:\n"
            "1 = issue an alert\n"
            "0 = do not issue an alert\n"
            "Never recommend an alert if the remaining alert budget is 0. Respone in JSON format. For example: {'action': 1}"
        )

    @property
    def example_rules(self) -> List[str]:
        example1 = (
            '{"background": "Heat alerts are not effective in cold weather", '
            '"rule": "If the heat index is below 80 F, do not issue an alert", '
            '"state relevance": "The current heat index is 78 F, which is below the threshold", '
            '"goal relevance": "The decision-makers goal is to avoid issuing unnecessary alerts if they are not effective"}'
        )

        example2 = (
            '{"background": "Repeated alerts can lead to alert fatigue", '
            '"rule": "If there have been 3 or more alerts in the last 7 days, do not issue an alert unless the heat index is above 102 F", '
            '"state relevance": "There have been 3 alerts in the last 7 days and the current heat index is 98 F, which is below the threshold", '
            '"goal relevance": "The decision-makers goal is to avoid alert fatigue by limiting the number of alerts issued"}'
        )

        example3 = (
            '{"background": "Heat alerts are more effective during weekends when people are more likely to be outdoors", '
            '"rule": "If it is a weekend and the heat index is above 90 F, issue an alert", '
            '"state relevance": "It is a weekend today and the current heat index is 92 F, which is above the threshold", '
            '"goal relevance": "The decision-makers goal is to issue alerts when they are most effective"}'
        )

        return [example1, example2, example3]

    def state_descriptor(self, *_, **__) -> str:
        """
        Convert the observation into a text description specific to the HeatAlerts environment.

        Returns:
            str: The text description of the observation.
        """
        template = (
            "\n- Location (FIPS code): {}"
            "\n- **Remaining alert budget**: {} "
            "\n- Current date and day of summer: {}"
            "\n- Current heat index: {} F"
            "\n- Quantile of current heat index relative to historical weather in current location (0-100): {} %"
            "\n- Average heat index over the past 7 days: {} F"
            "\n- Excess heat compared to the last 7 days: {} F"
            "\n- Weekend (yes/no): {} "
            "\n- Holiday (yes/no): {} "
            "\n- Alerts in last 14 days: {} "
            "\n- Alerts in last 7 days: {} "
            "\n- Alerts in last 3 days: {} "
            "\n- Alert streak: {} "
            "\n- Heat forecasts for next 14 days: {} "
            "\n- Max forecasted per week for the rest of the summer: {}"
        )
        env = self.env
        date = env.ep.index[env.t]
        obs = env.observation
        ep = env.ep

        # get hi max series
        hi_max = ep["hi_max"].values * 100  # this is in Fahrenheit
        curr_hi = hi_max[env.t]
        heat_qi = ep["heat_qi"].values * 100

        # last week
        t_ixs = np.arange(env.t - 6, env.t + 1)
        fm7 = hi_max[t_ixs]
        fm7_mean = fm7.mean()
        excess_fm7 = curr_hi - fm7_mean

        # get the forecast for next 14 days with some error growing per day
        # get the forecasted heat index for the next 14 days as a dict
        error = np.random.normal(0, 0.1, 14) * np.arange(1, 15)
        f14 = hi_max[env.t + 1 : env.t + 15] + error      
        # f14 = f14.round(2).astype(int).astype(str)
        dates = ep.index[env.t + 1 : env.t + 15]
        f14 = dict(zip(dates, f14))
        f14 = {k: f"{int(v)} F" for k, v in f14.items()}
        # f14 = (f14.round().astype(str) + "%").to_dict()
        # f14 = "\n  * ".join([f"{k}: {v}" for k, v in f14.items()])

        # forecast per remaining weeks
        ep_ = ep.iloc[env.t + 1 :].copy()
        ep_["week"] = ep_.index.str.slice(0, 7)
        hi_max_weekly = ep_.groupby("week")["hi_max"].max()
        hi_max_weekly = (
            (100 * hi_max_weekly).round(2).astype(int).astype(str) + " F"
        ).to_dict()

        obs_text = template.format(
            env.location,
            int(obs.remaining_budget) if obs.remaining_budget > 0 else "0 (no alerts left)",
            date,
            int(curr_hi),
            int(heat_qi[env.t]),
            int(fm7_mean),
            f"{'+' if excess_fm7 > 0 else ''}{int(excess_fm7)}",
            "yes" if obs.weekend else "no",
            "yes" if obs.holiday else "no",
            sum(env.actual_alert_buffer[-14:]) if env.t > 1 else 0,
            sum(env.actual_alert_buffer[-7:]) if env.t > 1 else 0,
            sum(env.actual_alert_buffer[-3:]) if env.t > 1 else 0,
            env.alert_streak,
            f14,
            hi_max_weekly,
        )

        return obs_text
