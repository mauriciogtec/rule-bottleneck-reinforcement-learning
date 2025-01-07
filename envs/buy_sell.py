from typing import Dict, Optional
import numpy as np
from gymnasium import Env, spaces
from langchain_together import Together, TogetherEmbeddings
import re
from datetime import date, timedelta

from envs.language_wrapper import LanguageWrapper


class BuySellText(Env):
    """In this environment, the agent must decide whether to buy, sell, or hold a stock.
    It can only do so once during each episode.
    Action = 0 -> Buy
    Action = 1 -> Hold
    Action = 2 -> Sell
    Budget = 1  # The agent starts with a budget of 1, after buying goes to 0, game ends when selling
                # forced to sell when budget is 0
    An LLM based generator creates "news" about the stock that can be used to make decisions.
    The transition in this version of the environment is based on the LLM.
    """

    init_outlook_prompt = (
        "## Task\n\nSimulates a the financial outlook of a popular stock of a tech company with ticker TEC. "
        "The current date is 2022-04-01. "
        "You answer should be a single short paragraph of one to two sentences without additional explanation o rnotes."
        "\n\n## Example answers:"
        "\n\n- The stock TEC will announce a new produce in 2022-04-05 and report earnings on 2022-04-08."
        "The stock is expected to rise after the announcement if the product is well received."
        "However, the earnings are expected to be below expectations."
        "\n\n- The stock TEC is expected to annoince a new product in 2022-04-05. The stock is expected to rise after the announcement."
        "\n\n- The stock TEC is expected to announce earnings on 2022-04-08. The stock is expected to fall after the announcement."
        "\n\n ## Your answer: "
    )

    init_price_prompt = (
        "## Task\n\nPredict the a week of stock prices for the stock TEC."
        "You will be given the initial outlook of the stock as of 2022-04-01."
        "Do not provide any additional information in your answer, only a list (JSON format) starting with ```[ and ending with ]```. "
        "Use two decimal places for the prices."
        "\n\n## Example answers:"
        "\n\n-  The price is ```[0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51]```"
        "\n\n ## Your answer: The price is "
    )

    news_prompt = (
        "## Task\n\nGiven a history of news about the stock since the the outlook "
        "Simulate the next-day news of a stock. "
        "You answer should be a single short paragraph of one to two sentences without additional explanation."
        "\n\n### Example answers:"
        "\n\n- The stock TEC has been performing well in the market. "
        "The company has announced a new product that is expected to increase the stock price."
        "\n\n- The stock TEC announced earnings that were below expectations. The stock price is expected to fall."
        "\n\n ## Your answer:\n"
    )

    next_price_template = (
        "## Task\n\nPredict the next day price of the stock TEC. "
        "You will be given the stock prices for the last 7 days and recent news about the stock. "
        "Do not provide any additional information in your answer, only a list (JSON format) starting with ``` and ending with ```."
        "Use two decimal places for the prices."
        "\n\nExample answers:"
        "\n\n- ```0.23```"
        "\n\n- ```0.45```"
        "\n\n## Initial outlook as of 2022-04-01\n\n {}"
        "\n\n## News\n\n {}"
        "\n\n## Prices over the last 7 days\n\n{}"
        "\n\n## Your answer:\n\nThe price is: "
    )

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

    def __init__(self):
        self.llm = Together(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")
        self.emb = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
        self.action_space = spaces.Discrete(3)

        # 1 for budget, 1 for buying price (0 otherwise) the last 7 prices, and 768 for the embeddings
        self.current_budget = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9 + 768,))

        raise NotImplementedError("This environment is not yet implemented")

    def reset(self, seed=None, options=None):
        # note, seed and options are not used
        # but are there for compatibility with the gym interface

        self.current_budget = 1
        self.current_date = date(2022, 4, 1)
        self.news = []
        self.buying_price = 0
        self.buying_date = None

        # get initial outlook
        self.initial_outlook = self.llm.invoke(
            self.init_outlook_prompt, max_tokens=100, temperature=0.5
        )
        self.initial_outlook = self.initial_outlook.strip().strip("\n")
        self.init_prices = self.llm.invoke(self.init_price_prompt, max_tokens=100)
        self.init_prices = self.init_prices.strip().strip("\n")

        # parse prices from the response
        try:
            self.init_prices = re.findall(r"\d+\.\d+", self.init_prices)
            self.init_prices = [float(price) for price in self.init_prices]

            # make sure it's the right length (seven) other wise crop or pad with last price
            if len(self.init_prices) < 7:
                self.init_prices = self.init_prices + [self.init_prices[-1]] * (
                    7 - len(self.init_prices)
                )
            elif len(self.init_prices) > 7:
                self.init_prices = self.init_prices[:7]
        except:
            self.init_prices = np.linspace(np.random.rand(), np.random.rand(), 8).round(
                2
            )

        text_obs = self.state_template.format(
            self.initial_outlook,
            self.current_date,
            self.init_prices,
            self.current_budget,
        )

        self.prices = np.array(self.init_prices)

        info = {
            "initial_outlook": self.initial_outlook,
            "initial_prices": self.init_prices,
            "has_bought": False,
            "buying_price": self.buying_price,
            "buying_date": self.buying_date,
            "current_date": self.current_date,
            "last_prices": self.prices,
            "text_obs": text_obs,
        }
        emb = self.emb.embed_query(text_obs)

        state = np.array(
            [self.current_budget, self.buying_price] + list(self.prices) + emb
        )

        return state, info

    def step(self, action):
        # advance the date
        self.current_date = self.current_date + timedelta(days=1)

        # simulate news
        news = self.llm.invoke(self.news_prompt, max_tokens=100).strip().strip("\n")
        self.news.append(news)

        # simulate next day price
        next_price_prompt = self.next_price_template.format(
            self.initial_outlook, "\n".join(self.news), self.prices
        )
        try:
            next_price = self.llm.invoke(next_price_prompt, max_tokens=10)
            # extract the first price
            next_price = re.findall(r"\d+\.\d+", next_price)[0]
        except:
            # if the model fails to predict the previous price
            # we just use the last price
            next_price = max(
                0.01, self.prices[-1] + 0.01 * np.round(np.random.randn(), 2)
            )

        # update the prices
        self.prices = np.append(self.prices[1:], float(next_price))

        # update the state
        if action == 0:
            # Buy
            self.current_budget = 0
            terminated = False
            self.buying_price = self.prices[-1]
            self.buying_date = self.current_date
            reward = 0
        elif action == 1:
            # Hold
            terminated = False
            # penalize holding when not bought
            if self.current_budget == 9:
                reward = -0.1
            else:
                reward = 0
        elif action == 2:
            # Sell
            if self.current_budget == 1:
                terminated = True
                reward = self.prices[-1] - self.buying_price
            else:
                terminated = False
                reward = -0.1
        else:
            raise ValueError("Invalid action")

        text_obs = self.state_template.format(
            self.initial_outlook,
            self.current_date,
            self.prices,
            self.current_budget,
        )

        info = {
            "initial_outlook": self.initial_outlook,
            "initial_prices": self.init_prices,
            "current_date": self.current_date,
            "last_prices": self.prices,
            "buying_date": self.buying_date,
            "buying_price": self.buying_price,
            "has_bought": self.current_budget == 0,
            "text_obs": text_obs,
        }

        emb = self.emb.embed_query(text_obs)

        state = np.array(
            [self.current_budget, self.buying_price] + list(self.prices) + emb
        )

        truncated = False

        return state, reward, terminated, truncated, info


if __name__ == "__main__":
    import sys  # not needed, just to stay within tradition of successful runs ending with 0
    from envs.language_wrapper import FinanceWrapper

    env = BuySellText()
    # step, info = env.reset()
    # state1, reward1, terminated1, truncated1, info1 = env.step(0)
    # state2, reward2, terminated2, truncated2, info2 = env.step(1)
    # print(info2)
    # print(f"Reward: {reward2}")

    wrapped_env = FinanceWrapper(env, env.emb)
    obs, info = wrapped_env.reset()
    print("Shape of observation: ", obs.shape)
    state1, reward1, terminated1, truncated1, info1 = wrapped_env.step(0)
    state2, reward2, terminated2, truncated2, info2 = wrapped_env.step(2)
    state3, reward3, terminated3, truncated3, info3 = wrapped_env.step(1)
    print(info3)
    print(f"Reward: {reward3}")

    sys.exit(0)


class BuySellTextLang(LanguageWrapper):
    """
    A wrapper for the Finance environment.
    """

    def __init__(self, **kwargs):
        env = BuySellText(**kwargs)
        super().__init__(env)

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
            " when to buy or sell a single stock. You will determine the action by"
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


class BuySellSimple(Env):
    """In this environment, the agent must decide whether to buy, sell, or hold a stock.
    It can only do so once during each episode.
    Action = 0 -> Buy
    Action = 1 -> Hold (wait to buy or sell)
    Action = 2 -> Sell

    Budget = 1  # The agent starts with a budget of 1, after buying goes to 0, game ends when selling
                # forced to sell when budget is 0

    The agent observes the drift and volatility, the current price, and indicator of having bought, and the buying price if available.
    Because of the operation delays and spread, the buy/sell price can be slightly different from the current price. Thus, 
    pay attention to the drift and volatility to make the decision.
    """

    def __init__(self, penalty=0.1):
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,))
        self.rng = np.random.default_rng()
        self.penalty = penalty

    def reset(self, seed: Optional[int] = None, options: Dict = {}):
        self.price = options.get("price", np.random.uniform(0.5, 1.5))
        self.volatility = options.get("volatility", 0.1 * self.price)
        self.drift = options.get("drift", np.random.uniform(-0.1, 0.1))

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_budget = 1.0
        self.buying_price = 0.0

        state = np.array(
            [
                self.drift,
                self.volatility,
                self.price,
                self.current_budget,
                self.buying_price,
            ],
            dtype=np.float32,
        )

        return state, {}

    def _update_drift_and_volatility(self):
        u = self.rng.random()
        if u < 0.25:
            self.volatility *= 0.75
        elif u < 0.5:
            self.volatility *= 1.5

        u = self.rng.random()
        if u < 0.25:
            self.drift -= self.volatility * np.random.rand()
        elif u < 0.5:
            self.drift += self.volatility * np.random.rand()

    def _update_price(self):
        self.price = self.price * np.exp(
            self.drift + self.volatility * self.rng.normal()
        )

    def step(self, action):
        self._update_price()
        self._update_drift_and_volatility()
        truncated = False

        if action == 0:  # buy
            # Buy
            done = False
            if self.current_budget == 1:
                self.current_budget = 0
                self.buying_price = self.price
                reward = 0
            else:
                reward = -self.penalty
        elif action == 1:  # hold/wait
            # Hold
            done = False
            reward = 0.0
        elif action == 2:  # sell
            # Sell
            if self.current_budget == 0:
                reward = self.price - self.buying_price
                done = True
            else:
                reward = -self.penalty
                done = False
        else:
            raise ValueError("Invalid action")

        state = np.array(
            [
                self.drift,
                self.volatility,
                self.price,
                self.current_budget,
                self.buying_price,
            ],
            dtype=np.float32,
        )

        return state, reward, done, truncated, {}


class BuySellSimpleLang(LanguageWrapper):
    """
    A wrapper for the Finance environment.
    """

    def __init__(self):
        env = BuySellSimple()
        super().__init__(env)

    @property
    def task_text(self) -> str:
        return (
            "You are an agent assisting a financial analyst in making optimized decisions about"
            " when to buy or sell a single stock. You will determine the action by"
            " considering the current stock price, the stock price history, and the analyst's predictions."
            " It is known that the stock price follows a geometric Brownian motion of the form:\n"
            " price(t+1) = price(t) * np.exp(drift + volatility * Z)\n"
            " where Z is a standard normal random variable."
            "\n\nYou will be given the current estimated drift and volatility of the stock, the current price, and an indicator"
            " of whether the stock has been bought or not, and at which price. "
            " The value of the drift and volatility at the decision time is not known, you only known the analyst's estimates."
            " But it is known that it follows a random walk with small deviations from the current estimated values."
            " Your reward is obtained at selling time, and it is the difference between the selling price and the buying price."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A single integer value representing the decision:\n"
            "0 = buy\n"
            "1 = hold/wait\n"
            "2 = sell\n"
            "You can only buy the stock if not bought yet, and you can only sell if it is already in the portfolio."
            # " You will be penalized if selecting to sell the stock when it has not been bought"
            # " or if selecting to buy the stock when it has already been bought."
        )

    def state_descriptor(self, obs, _):
        """
        Convert the observation into a text description specific to the Finance environment.

        Args:
            obs (Any): The observation to convert into text.
            info (dict[str, Any]): Additional information about the observation.

        Returns:
            str: The text description of the observation.
        """

        drift = obs[0]
        volatility = obs[1]
        current_price = obs[2]
        current_budget = obs[3]
        buying_price = obs[4]

        # First the current price
        text = f"Current price: ${current_price:.2f}\n"

        # Then the estimated drift and volatility
        text += f"Estimated drift: {drift:.2f}\n"
        text += f"Estimated volatility: {volatility:.2f}\n"

        # Is the stock currently in the portfolio? If so, what was the buying price?
        if current_budget == 0:
            text += f"Stock in portfolio. Bought at ${buying_price:.2f}\n"
        else:
            text += "Stock not in portfolio."

        return text
