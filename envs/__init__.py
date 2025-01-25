from functools import partial

from gymnasium.envs.registration import register
from weather2alert.env import HeatAlertEnv

from envs.buy_sell import BuySellSimple, BuySellSimpleLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsSimple, VitalSignsSimpleLang

from . import wrappers

Uganda = partial(VitalSignsSimpleLang, "models/uganda.npz", time_discount=0.95)
MimicIII = partial(VitalSignsSimpleLang, "models/mimic-iii.npz", time_discount=0.95)
MimicIV = partial(VitalSignsSimpleLang, "models/mimic-iv.npz", time_discount=0.95)
HeatAlerts = partial(HeatAlertsLang)
BuySellSimple = partial(BuySellSimpleLang)

UgandaNumeric = partial(VitalSignsSimple, "models/uganda.npz", time_discount=0.95)
MimicIIINumeric = partial(VitalSignsSimple, "models/mimic-iii.npz", time_discount=0.95)
MimicIVNumeric = partial(VitalSignsSimple, "models/mimic-iv.npz", time_discount=0.95)
HeatAlertsNumeric = partial(HeatAlertEnv, budget=10, sample_budget=True, random_starts=True)
BuySellSimpleNumeric = partial(BuySellSimple)


kwargs = {"disable_env_checker": True}
register(id="Uganda", entry_point="envs:Uganda", **kwargs)
register(id="MimicIII", entry_point="envs:MimicIII", **kwargs)
register(id="MimicIV", entry_point="envs:MimicIV", **kwargs)
register(id="BuySellSimple", entry_point="envs:BuySellSimple", **kwargs)
register(id="HeatAlerts", entry_point="envs:HeatAlerts", **kwargs)

register(id="UgandaNumeric", entry_point="envs:UgandaNumeric", **kwargs)
register(id="MimicIIINumeric", entry_point="envs:MimicIIINumeric", **kwargs)
register(id="MimicIVNumeric", entry_point="envs:MimicIVNumeric", **kwargs)
register(id="BuySellSimpleNumeric", entry_point="envs:BuySellSimpleNumeric", **kwargs)
register(id="HeatAlertsNumeric", entry_point="envs:HeatAlertsNumeric", **kwargs)
