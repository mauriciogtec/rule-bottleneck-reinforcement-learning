from envs.finance import BuySellHoldLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsLang


if __name__ == "__main__":
    import sys

    env1 = BuySellHoldLang()
    print(env1.task_text)

    env2 = HeatAlertsLang()
    print(env2.task_text)

    env3 = VitalSignsLang("models/uganda.npz")
    print(env3.task_text)

    sys.exit(0)
