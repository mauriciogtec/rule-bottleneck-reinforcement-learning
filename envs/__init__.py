from envs.buy_sell import BuySellTextLang, BuySellLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsLang
from gymnasium.envs.registration import register
from functools import partial

Uganda = partial(VitalSignsLang, "models/uganda.npz")
MimicIII = partial(VitalSignsLang, "models/mimic-iii.npz")
MimicIV = partial(VitalSignsLang, "models/mimic-iv.npz")
HeatAlerts = partial(HeatAlertsLang)
BuySellText = partial(BuySellTextLang)
BuySell2 = partial(BuySellLang)


kwargs = {"disable_env_checker": True}
register(id="Uganda", entry_point="envs:Uganda", **kwargs)
register(id="MimicIII", entry_point="envs:MimicIII", **kwargs)
register(id="MimicIV", entry_point="envs:MimicIV", **kwargs)
register(id="HeatAlerts", entry_point="envs:HeatAlerts", **kwargs)
register(id="BuySellText", entry_point="envs:BuySellText", max_episode_steps=16, **kwargs)
register(id="BuySell", entry_point="envs:BuySell2", max_episode_steps=16, **kwargs)
