import numpy as np
from gymnasium import Env, spaces
from langchain_together import Together, TogetherEmbeddings
import re
from datetime import date, timedelta


class BuySellHold(Env):
    """In this environment, the agent must decide whether to buy, sell, or hold a stock.
    It can only do so once during each episode.
    Action = 0 -> Buy
    Action = 1 -> Sell
    Action = 2 -> Hold
    Budget = 1  # The agent starts with a budget of 1, after buying goes to 0, game ends when selling
                # forced to sell when budget is 0
    An LLM based generator creates "news" about the stock that can be used to make decisions.
    """

    init_outlook_prompt = (
        "### Task\n\nSimulates a the financial outlook of a popular stock of a tech company with ticker TEC. "
        "The current date is 2022-04-01. "
        "You answer should be a single short paragraph of one to two sentences without additional explanation o rnotes."
        "\n\n### Example answers:"
        "\n\n- The stock TEC will announce a new produce in 2022-04-05 and report earnings on 2022-04-08."
        "The stock is expected to rise after the announcement if the product is well received."
        "However, the earnings are expected to be below expectations."
        "\n\n- The stock TEC is expected to annoince a new product in 2022-04-05. The stock is expected to rise after the announcement."
        "\n\n- The stock TEC is expected to announce earnings on 2022-04-08. The stock is expected to fall after the announcement."
        "\n\n ## Your answer: The "
    )

    init_price_prompt = (
        "## Task\n\nPredict the a week of stock prices for the stock TEC."
        "You will be given the initial outlook of the stock as of 2022-04-01."
        "Do not provide any additional information in your answer, only a list (JSON format) starting with ```json\n[ and ending with ]```. "
        "Use two decimal places for the prices."
        "\n\nExample answers:"
        "\n\n- ```json\n[0.23, 0.25, 0.26, 0.24, 0.27, 0.28, 0.29]\n```"
        "\n\n- ```json\n[0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51]\n```"
    )

    news_prompt = (
        "## Task\n\nGiven a history of news about the stock since the the outlook "
        "Simulate the next-day news of a stock. "
        "You answer should be a single short paragraph of one to two sentences without additional explanation."
        "\n\n### Example answers:"
        "\n\n- The stock TEC has been performing well in the market. "
        "The company has announced a new product that is expected to increase the stock price."
        "\n\n- The stock TEC announced earnings that were below expectations. The stock price is expected to fall."
        "\n\n ## Your answer: The "
    )

    next_price_template = (
        "## Task\n\nPredict the next day price of the stock TEC. "
        "You will be given the stock prices for the last 7 days and recent news about the stock. "
        "Do not provide any additional information in your answer, only a list (JSON format) starting with ```json\n and ending with ```."
        "Use two decimal places for the prices."
        "\n\nExample answers:"
        "\n\n- ```json\n0.23\n```"
        "\n\n- ```json\n0.45\n```"
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
        "## Has bought?\n"
        "{}"
    )

    def __init__(self):
        self.llm = Together(model="meta-llama/Llama-3.2-3B-Instruct-Turbo")
        self.emb = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
        self.action_space = spaces.Discrete(3)

        # 1 for budget, the last 8 prices, and 768 for the embeddings
        self.current_budget = 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9 + 768,))

    def reset(self):
        self.current_budget = 1
        self.current_date = date(2022, 4, 1)
        self.news = []

        # get initial outlook
        self.initial_outlook = (
            self.llm.invoke(self.init_outlook_prompt, max_tokens=100).strip().strip("\n")
        )
        self.init_prices = self.llm.invoke(self.init_price_prompt, max_tokens=100).strip().strip("\n")

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
            self.init_prices = np.linspace(np.random.rand(), np.random.rand(), 8).round(2)

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
            "last_prices": self.prices,
            "text_obs": text_obs,
        }
        emb = self.emb.embed_query(text_obs)

        state = np.array([self.current_budget] + list(self.prices) + emb)

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
            next_price = max(0.01, self.prices[-1] + 0.01 * np.round(np.random.randn(), 2))

        # update the prices
        self.prices = np.append(self.prices[1:], float(next_price))

        # update the state
        if action == 0:
            # Buy
            self.current_budget = 0
            terminated = False
            self.buying_price = self.prices[-1]
            reward = 0
        elif action == 1:
            # Sell
            terminated = True
            reward = self.prices[-1] - self.buying_price
        else:
            # Hold
            terminated = False
            reward = 0

        text_obs = self.state_template.format(
            self.initial_outlook,
            self.current_date,
            self.prices,
            self.current_budget,
        )

        info = {
            "initial_outlook": self.initial_outlook,
            "initial_prices": self.init_prices,
            "last_prices": self.prices,
            "text_obs": text_obs,
        }

        emb = self.emb.embed_query(text_obs)

        state = np.array([self.current_budget] + list(self.prices) + emb)

        truncated = False

        return state, reward, terminated, truncated, info


if __name__ == "__main__":
    import sys  # not needed, just to stay within tradition of successful runs ending with 0

    env = BuySellHold()
    step, info = env.reset()
    state1, reward1, terminated1, truncated1, info1 = env.step(0)
    state2, reward2, terminated2, truncated2, info2 = env.step(1)
    sys.exit(0)
