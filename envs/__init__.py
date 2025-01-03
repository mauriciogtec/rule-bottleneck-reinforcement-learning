from envs.finance import BuySellHoldLang
from envs.heat_alerts import HeatAlertsLang
from envs.vital_signs import VitalSignsLang
from gymnasium.envs.registration import register

uganda = lambda: VitalSignsLang("models/uganda.npz")
mimic_iii = lambda: VitalSignsLang("models/mimic-iii.npz")
mimic_iv = lambda: VitalSignsLang("models/mimic-iv.npz")
heat_alerts = lambda: HeatAlertsLang()
buy_sell_hold = lambda: BuySellHoldLang()

register(id="uganda", entry_point="envs:uganda")
register(id="mimic-iii", entry_point="envs:mimic_iii")
register(id="mimic-iv", entry_point="envs:mimi_iv")
register(id="buy_sell_hold", entry_point="envs:BuySellHoldLang")
register(id="heat_alerts", entry_point="envs:HeatAlertsLang")
