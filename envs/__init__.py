from functools import partial

from gymnasium.envs.registration import register
from weather2alert.env import HeatAlertEnv

from envs.buy_sell import BuySellSimple, BuySellSimpleLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsSimple, VitalSignsSimpleLang
from envs.bin_packing import (
    BinPacking as BinPackingNumeric,
    BinPackingIncremental as BinPackingIncrementalNumeric,
    BinPackingLang,
    BinPackingIncrementalLang,
)

import gymnasium as gym

Uganda = partial(VitalSignsSimpleLang, "models/uganda.npz", time_discount=0.95)
MimicIII = partial(VitalSignsSimpleLang, "models/mimic-iii.npz", time_discount=0.95)
MimicIV = partial(VitalSignsSimpleLang, "models/mimic-iv.npz", time_discount=0.95)
HeatAlerts = partial(
    HeatAlertsLang,
    budget=10,
    sample_budget=False,
    effectiveness_type="data",
    reward_type="saved",
    random_starts=True,
    penalty=0.1,
    top_k_fips=10,
)
BuySellSimple = partial(BuySellSimpleLang)
BinPacking = partial(BinPackingLang)
BinPackingIncremental = partial(BinPackingIncrementalLang)

UgandaNumeric = partial(VitalSignsSimple, "models/uganda.npz", time_discount=0.95)
MimicIIINumeric = partial(VitalSignsSimple, "models/mimic-iii.npz", time_discount=0.95)
MimicIVNumeric = partial(VitalSignsSimple, "models/mimic-iv.npz", time_discount=0.95)
HeatAlertsNumeric = partial(
    HeatAlertEnv,
    budget=10,
    sample_budget=False,
    effectiveness_type="data",
    reward_type="saved",
    random_starts=True,
    penalty=0.1,
    top_k_fips=10,
)
BuySellSimpleNumeric = partial(BuySellSimple)
BinPackingNumeric = partial(BinPackingNumeric)
BinPackingIncrementalNumeric = partial(BinPackingIncrementalNumeric)


kwargs = {"disable_env_checker": True}

register(id="Uganda", entry_point="envs:Uganda", **kwargs)
register(id="MimicIII", entry_point="envs:MimicIII", **kwargs)
register(id="MimicIV", entry_point="envs:MimicIV", **kwargs)
register(id="BuySellSimple", entry_point="envs:BuySellSimple", **kwargs)
register(id="BinPacking", entry_point="envs:BinPacking", **kwargs)
register(id="BinPackingIncremental", entry_point="envs:BinPackingIncremental", **kwargs)
register(
    id="HeatAlerts",
    entry_point="envs:HeatAlerts",
    # additional_wrappers=[reward_scaler],
    **kwargs
)

register(id="UgandaNumeric", entry_point="envs:UgandaNumeric", **kwargs)
register(id="MimicIIINumeric", entry_point="envs:MimicIIINumeric", **kwargs)
register(id="MimicIVNumeric", entry_point="envs:MimicIVNumeric", **kwargs)
register(id="BuySellSimpleNumeric", entry_point="envs:BuySellSimpleNumeric", **kwargs)
register(id="BinPackingNumeric", entry_point="envs:BinPackingNumeric", **kwargs)
register(
    id="BinPackingIncrementalNumeric",
    entry_point="envs:BinPackingIncrementalNumeric",
    **kwargs
)
register(
    id="HeatAlertsNumeric",
    entry_point="envs:HeatAlertsNumeric",
    # additional_wrappers=[reward_scaler],
    **kwargs
)
