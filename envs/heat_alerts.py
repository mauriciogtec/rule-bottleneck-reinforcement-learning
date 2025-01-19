from typing import List, Optional

from envs.wrappers import LanguageWrapper


class HeatAlertsLang(LanguageWrapper):
    """
    A wrapper for the HeatAlerts environment from Considine et al. (2024).
    """

    def __init__(
        self,
        budget: Optional[int] = None,
        fips_list: Optional[List[int]] = None,
        years: Optional[List[int]] = None,
    ):
        try:
            from weather2alert.env import HeatAlertEnv
        except ImportError:
            raise ImportError(
                "The HeatAlertsWrapper requires the weather2alert package to be installed."
                " Please install it using `pip install git+https://github.com/NSAPH-Projects/weather2alert@dev`."
            )
        env = HeatAlertEnv(budget=budget, fips_list=fips_list, years=years)
        super().__init__(env)

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
