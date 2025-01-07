from envs.buy_sell import BuySellSimpleLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsSimpleLang
from gymnasium.envs.registration import register
from functools import partial

Uganda = partial(VitalSignsSimpleLang, "models/uganda.npz")
MimicIII = partial(VitalSignsSimpleLang, "models/mimic-iii.npz")
MimicIV = partial(VitalSignsSimpleLang, "models/mimic-iv.npz")
HeatAlerts = partial(HeatAlertsLang)
BuySellSimple = partial(BuySellSimpleLang)


# kwargs = {"disable_env_checker": True, "order_enforce": False}
kwargs = {}
register(id="Uganda", entry_point="envs:Uganda", **kwargs)
register(id="MimicIII", entry_point="envs:MimicIII", **kwargs)
register(id="MimicIV", entry_point="envs:MimicIV", **kwargs)
register(id="HeatAlerts", entry_point="envs:HeatAlerts", **kwargs)
register(id="BuySellSimple", entry_point="envs:BuySellSimple", max_episode_steps=16, **kwargs)
