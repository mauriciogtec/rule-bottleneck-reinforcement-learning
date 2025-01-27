import math
from typing import List, Optional

import numpy as np
from gymnasium import Env, spaces
from sklearn.mixture import GaussianMixture

from envs.wrappers import LanguageWrapper


def temperature_penalty(temperature):
    if temperature <= 38:
        return 0
    else:
        return -math.exp(abs(temperature - 38.0) / 2)  # Exponential penalty


def pulse_penalty(pulse):
    if pulse <= 120:
        return 0
    else:
        return -math.exp(abs(pulse - 120) / 17)  # Exponential penalty


def respiratory_penalty(respiratory_rate):
    if respiratory_rate <= 30:
        return 0
    else:
        return -math.exp(abs(respiratory_rate - 30) / 5)  # Exponential penalty


def spo2_penalty(spo2):
    # if 90 <= spo2: # HUGE BUG!!!
    if spo2 >= 90:
        return 0
    else:
        return -math.exp(abs(spo2 - 90) / 4)  # Exponential penalty


def blood_penalty(blood_pressure):
    if blood_pressure <= 127:
        return 0
    else:
        return -math.exp(abs(blood_pressure - 127) / 5)  # Exponential penalty


def reward_function(sign_dict, min_max, clip_value=1, scaler=0.1):
    reward = 0
    sign_dict = {
        sign: c * (min_max[sign][1] - min_max[sign][0]) + min_max[sign][0]
        for sign, c in sign_dict.items()
    }
    for signs in sign_dict:
        if signs == "COVERED_SKIN_TEMPERATURE":
            reward += temperature_penalty(sign_dict[signs])
        elif signs == "PULSE_RATE":
            reward += pulse_penalty(sign_dict[signs])
        elif signs == "RESPIRATORY_RATE":
            reward += respiratory_penalty(sign_dict[signs])
        elif signs == "SPO2":
            reward += spo2_penalty(sign_dict[signs])

    # Scale and clip the reward to avoid large values
    reward = reward * scaler
    reward = np.maximum(-clip_value, reward)

    return reward


class VitalSignsSimple(Env):

    def __init__(
        self,
        path: str,
        init_agents=4,  # B=3 in the paper
        # max_num_agents=10,  # N=20 in the paper
        budget=5,  # They have a budget, which does not necessarily eexample_rulesl to init_agent
        # t_min=1,  # t_min = 3 in the paper
        # t_max=5,  # t_max = 5 in the paper
        system_duration=10,  # = 50 in the paper, no letter
        degree_of_arm_noise=0.15,
        intervention_success_rate=0.7,
        variability_window=5,
        # joining_number=2,  # Here, vital signs only advance after N patients join
        joining_interval=5,  # Here, simulate the number of internal vital signs steps
        T: Optional[int] = None,  # planning length / for finite horizon evaluation
        time_discount: Optional[float] = 0.99,  # discount factor for time,
        ignore_free_penalty: Optional[float] = 1.0,
    ):
        ## Check inputs
        assert (
            init_agents <= budget
        ), "The number of agents should be less than or equal to the budget"

        ## Random number generator
        self.np_random = np.random.default_rng()
        self.T = T
        self.ignore_free_penalty = ignore_free_penalty

        # load GMM
        self._load_gmm(path)

        ## We are in a finite time horizon
        # self.T = T
        # self.remaining_planning_length = T
        ## Number of agents at the initial timestep
        self.init_agents = init_agents
        self.num_agents = init_agents
        self.budget = budget
        # self.t_min = t_min
        # self.t_max = t_max
        # self.joining_number = joining_number
        self.system_duration = system_duration
        self.joining_interval = joining_interval

        self.degree_of_arm_noise = degree_of_arm_noise
        self.intervention_success_rate = intervention_success_rate
        self.variability_window = variability_window

        ## Actions is a list of index that denotes agents who are pulled
        # total_action = math.comb(max_num_agents, budget)
        self.action_space = spaces.Discrete(budget)

        # The space of each device needs:
        self.nv = len(self.vital_signs)
        # 1. how many steps the current holder has worn the device (1)
        # 2. the sign history of the current holder (variability_window * nv)
        # 2. vital signs mean and std. dev. of the current holder (2 * nv)

        self.per_device_dim = 1 + (self.variability_window + 2) * self.nv
        self.total_dim = self.budget * self.per_device_dim
        # self.observation_space = spaces.Box(0, 1.0, shape=(self.total_dim,))
        self.observation_space = spaces.Box(
            0, 1.0, shape=(self.budget, self.per_device_dim), dtype=np.float32
        )

        ## Track agent states
        self.device_states = [{} for _ in range(self.budget)]

        # Time discount for future reward computation.
        self.time_discount = time_discount

    def _initialize_agents(self, init_agents=None):
        """Initialize the agents' states at timestep 0"""
        assignments = set(np.random.choice(self.budget, init_agents, replace=False))
        for i in range(self.budget):
            if i in assignments:
                state, _, mean, cov = self._sample_agent()
                current_vital, variability, signs_history = state

                # Update the device states
                self.device_states[i]["time_worn"] = 1
                self.device_states[i]["vitals"] = current_vital
                self.device_states[i]["variability"] = variability
                self.device_states[i]["signs_history"] = signs_history
                self.device_states[i]["gmm"] = mean, cov
            else:
                self.device_states[i]["time_worn"] = 0
                self.device_states[i]["vitals"] = np.zeros(self.nv)
                self.device_states[i]["variability"] = np.zeros(self.nv)
                sign_history = np.zeros((self.nv, self.variability_window))
                self.device_states[i]["signs_history"] = sign_history

    def _state_to_obs(self):
        # Initialize an empty list to hold rows
        agent_matrix = np.zeros((self.budget, self.per_device_dim), dtype=np.float32)

        # Iterate over each agent's state
        for j, agent in enumerate(self.device_states):
            # Extract values from the dictionary
            time_worn = agent["time_worn"] / self.system_duration

            # vitals = list(agent["vitals"])  # nv elements, ignore, part of history
            signs_history = np.array(agent["signs_history"], dtype=np.float32)
            mean, std = signs_history.mean(axis=1), signs_history.std(axis=1)

            # Concatenate all components into a single row
            agent_matrix[j, :] = np.concatenate(
                [[time_worn], signs_history.flatten(), mean, std]
            )

        return agent_matrix  # .flatten()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        # Reset time counter
        self.t = 0

        # Empty the device states

        # Initialize agents at timestep 0
        options = options if options is not None else {}
        init_agents = options.get("init_agents", self.init_agents)
        self._initialize_agents(init_agents=init_agents)

        # # Set the remaining planning length
        # self.remaining_planning_length = self.T

        obs = self._state_to_obs()
        info = {}

        return obs, info

    def step(self, action: int):
        """

        Args:
            action (int): This is the device that we want to assign to a new patient
        """
        self.t += 1
        # 1. Assign the device, if the device has a current holder. If it does, compute the
        #    remaining reward/cost by simulating the rest of their time in the system.
        #    if the device is free, no reward, just assign the device to the new patient
        # 2. Update the device/agent states

        # == 1. Device assignment ==

        # Set the remaining planning length
        # self.remaining_planning_length -= 1

        # This will be an infinite horizon problem, and we will let gym
        # handle the time limit, which will set truncated to True when max steps reached
        terminated = (self.t >= self.T) if self.T else False
        truncated = False

        # Sample the new patient
        state, _, mean, cov = self._sample_agent()
        current_vital, variability, signs_history = state

        is_free = self.device_states[action]["time_worn"] == 0
        reward = 0.0

        gamma = self.time_discount

        num_free = sum([self.device_states[i]["time_worn"] == 0 for i in range(self.budget)])
        if not is_free:
            # first check that there was no remaining free device, otherwise penalzie
            if num_free > 0:
                reward -= self.ignore_free_penalty

            # simulate the rest of the time for the current holder
            mean, cov = self.device_states[action]["gmm"]
            time_worn = self.device_states[action]["time_worn"]

            if self.T is not None:
                remaining = min(self.T - self.t, self.system_duration - time_worn)
            else:
                remaining = self.system_duration - time_worn

            for t in range(remaining):
                # device is free, simulate remaining time without device
                state, r = self._simulate_one_step(action, intervention=False)

                # update agent state
                self.device_states[action]["time_worn"] += 1
                self.device_states[action]["vitals"] = current_vital
                self.device_states[action]["variability"] = variability
                self.device_states[action]["signs_history"] = signs_history

                reward += r * (gamma**t)

        # Assign the device to the new patient
        self.device_states[action]["time_worn"] = 0
        self.device_states[action]["vitals"] = current_vital
        self.device_states[action]["variability"] = variability
        self.device_states[action]["signs_history"] = signs_history
        self.device_states[action]["gmm"] = mean, cov

        # == 2. Update the remaining device/agents ==
        for i in range(self.budget):
            is_free = self.device_states[i]["time_worn"] == 0
            if not is_free or i == action:  # also compute for the new patient
                self.device_states[i]["time_worn"] += 1

                # advance the vital signs
                state, r = self._simulate_one_step(i, intervention=True)

                # update agent state
                self.device_states[i]["vitals"] = state[0]
                self.device_states[i]["variability"] = state[1]
                self.device_states[i]["signs_history"] = state[2]

                reward += r

                # remove them from the system if system reached
                if self.device_states[i]["time_worn"] >= self.system_duration:
                    self.device_states[i]["time_worn"] = 0
                    self.device_states[i]["vitals"] = np.zeros(self.nv)
                    self.device_states[i]["variability"] = np.zeros(self.nv)
                    sign_history = np.zeros((self.nv, self.variability_window))
                    self.device_states[i]["signs_history"] = sign_history

        obs = self._state_to_obs()

        if terminated:
            # for all devices compute the reward for the remaining time
            for i in range(self.budget):
                is_free = self.device_states[i]["time_worn"] == 0
                if not is_free:
                    mean, cov = self.device_states[i]["gmm"]
                    time_worn = self.device_states[i]["time_worn"]
                    gamma = self.time_discount
                    for t in range(time_worn, self.system_duration):
                        # device is free, simulate remaining time without device
                        state, r = self._simulate_one_step(i, intervention=True)

                        # update agent state
                        self.device_states[i]["time_worn"] += 1
                        self.device_states[i]["vitals"] = state[0]
                        self.device_states[i]["variability"] = state[1]
                        self.device_states[i]["signs_history"] = state[2]

                        reward += r * (gamma ** (t - time_worn))

        return obs, reward, terminated, truncated, {}

    def _conditional_sample_mnd(self, vital_values, mean, cov):
        """
        Sample from the conditional distribution of a multivariate Normal Distribution

        mean and cov were trained on trainsition pairs
        (st ; st+1) ~ MVN (m, S)
        then
        st+1 | st ~ MVN (m1, S1)
        m1 = m2 + S21 S-1 (st - m1)
        S1 = S22 - S21 S-1 S12
        """
        given_indices = np.arange(len(vital_values))
        remaining_indices = np.arange(len(vital_values), len(mean))

        # Calculate conditional means and covariances for each component
        mean_given = mean[given_indices]
        mean_remaining = mean[remaining_indices]
        cov_given_given = cov[np.ix_(given_indices, given_indices)]
        cov_remaining_given = cov[np.ix_(remaining_indices, given_indices)]
        cov_given_remaining = cov[np.ix_(given_indices, remaining_indices)]
        cov_remaining_remaining = cov[np.ix_(remaining_indices, remaining_indices)]
        # print("means",mean_given,mean_remaining)
        # print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

        cov_inv_given_given = np.linalg.inv(cov_given_given)
        conditional_mean = (
            mean_remaining
            + cov_remaining_given @ cov_inv_given_given @ (vital_values - mean_given)
        )
        conditional_cov = (
            cov_remaining_remaining
            - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining
        )

        v = self.np_random.multivariate_normal(
            mean=conditional_mean, cov=conditional_cov
        )

        return np.clip(v, 0, 1)

    # current_values, min_max, intervention_success_rate, mean=None, cov=None,
    def _interventions(self, vital_values, mean, cov):
        """interventions: This function models the effect of intervention. If the patient's value
        falls in the normal range, then the patient's next state will be sampled from a multivariate
        Guassian from this current state

        If the patient's vital sign shows abnormality, then there is a 30% chance the doctors do not
        intervene, and there is a 70% chance the intervention creates a positive effect on the patient.
        After applying the positive effect, the patient's new state will be the condition for sampling
        the next state
        """
        vital_signs = self.vital_signs
        min_max = self.min_max

        rew = reward_function(dict(zip(vital_signs, vital_values)), min_max)
        if rew >= 0:
            return self._conditional_sample_mnd(vital_values, mean, cov)
        else:
            # new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
            # print("Old", current_values)
            new_signs = self._improve_vital_signs3(
                dict(zip(vital_signs, vital_values)), mean, cov
            )
            # print("NEW",[new_signs[vital] for vital in vital_signs])
            return self._conditional_sample_mnd(
                [new_signs[vital] for vital in vital_signs], mean, cov
            )

    def _simulate_one_step(self, device_index, intervention=False):
        """simulate_one_step: based on the current value, calculate what's the next state for vital signs,
        the variance of vital sign for the past five timesteps, and the reward
        """

        vital_signs = self.vital_signs
        min_max = self.min_max

        mean, cov = self.device_states[device_index]["gmm"]
        vitals = self.device_states[device_index]["vitals"]
        signs_history = self.device_states[device_index]["signs_history"]
        variability = self.device_states[device_index]["variability"]

        if intervention:
            next_signs = self._interventions(vitals, mean, cov)
        else:
            next_signs = self._conditional_sample_mnd(vitals, mean, cov)

        for i in range(len(vital_signs)):
            signs_history[i].pop(0)
            signs_history[i].append(next_signs[i])

        # Note: Mauricio changed to standard deviation, better normalization
        variability = np.array([np.std(l) for l in signs_history])

        reward = reward_function(dict(zip(vital_signs, next_signs)), min_max)
        return [next_signs, variability, signs_history], reward

    def _load_gmm(self, path: str) -> GaussianMixture:
        data = np.load(path)
        means = np.array(data["means"])
        covariances = np.array(data["covariances"])
        weights = np.array(data["weights"])
        scaler_min = np.array(data["scaler_min"])
        scaler_max = np.array(data["scaler_max"])
        names = list((data["names"]))

        min_max = {
            name: [min_val, max_val]
            for name, min_val, max_val in zip(names, scaler_min, scaler_max)
        }

        # Reconstruct the GMM
        gmm = GaussianMixture(n_components=len(weights), covariance_type="full")

        # Manually set the parameters
        gmm.weights_ = weights
        gmm.means_ = means
        gmm.covariances_ = covariances

        # Compute the precisions_cholesky_ required by the GaussianMixture object
        gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

        self.gmm = gmm
        self.min_max = min_max
        self.vital_signs = names
        self.given_indices = np.arange(len(self.vital_signs))

    def _sample_agent(self):  # moves this to in class, all functions should be in class
        """sample_agent: you choose a component basesd on weight of each component for the multivariate
        Gaussian, then you get the sample from it.
        You perturb the vital sign mean and cov by choosing a mean and covariance from another component
        in the mixture model, and randomly sampling a influence factor to determine the magnitude of
        pertubation
        """
        gmm = self.gmm
        self.min_max = self.min_max

        weights = gmm.weights_

        # Normalize the weights to ensure they sum to 1
        weights /= np.sum(weights)

        # Sample an index based on the weights
        component = self.np_random.choice(len(weights), p=weights)

        means = gmm.means_
        covariances = gmm.covariances_
        mean = means[component]
        cov = covariances[component]
        state, _ = self._resample_values(mean, cov)

        perturb = self.np_random.choice(
            [i for i in range(len(weights)) if i != component]
        )

        x = self.np_random.uniform(0, self.degree_of_arm_noise)
        y = self.np_random.uniform(0, self.degree_of_arm_noise)

        mean = (1 - x) * mean + x * means[perturb]
        cov = (1 - y) * cov + y * covariances[perturb]

        # print(mean,cov)
        # pertubation
        return state, component, mean, cov

    def _resample_values(self, mean, cov):
        """resample_values: You sample from a multivariate Gaussian for your initial value,
        and you sample conditioned on the previous value until you have enough sign history to
        calculate variability

        Then you return the current signs, the variability of the past timesteps, the past
        vital sign values, and the corresponding reward of the currrent vital sign
        """
        variability_window = self.variability_window

        vital_signs = self.vital_signs
        min_max = self.min_max
        given_indices = np.arange(len(vital_signs))

        sample = self.np_random.multivariate_normal(mean=mean, cov=cov)
        sample = np.clip(sample, 0, 1)

        current_signs = [sample[i] for i in given_indices]
        signs_history = [[] for _ in range(len(vital_signs))]
        for i in range(len(vital_signs)):
            signs_history[i].append(sample[i])

        for _ in range(variability_window - 1):
            current_signs = self._conditional_sample_mnd(current_signs, mean, cov)
            for i in range(len(vital_signs)):
                signs_history[i].append(current_signs[i])

        # print(signs_history)
        # for l in signs_history:
        # print(l,np.var(l))
        variability = np.array([np.std(l) for l in signs_history])
        # print(variability)
        reward = reward_function(dict(zip(vital_signs, current_signs)), min_max)
        return [current_signs, variability, signs_history], reward

    # sign_dict, min_max, intervention_success_rate
    def _improve_vital_signs3(self, sign_dict, mean, cov):
        """improve_vital_signs: Another model of the positive effect of intervention
        (assigning a medical device). The medical staff reacts to the alert 70%
        of time in reality, and the abnormal vital sign is adjusted probablistically
        towards the normal. This seems to be the model used in the final paper.
        """

        min_max = self.min_max
        if min_max:
            # normalize
            sign_dict = {
                sign: c * (min_max[sign][1] - min_max[sign][0]) + min_max[sign][0]
                for sign, c in sign_dict.items()
            }

        # print(sign_dict)
        if self.np_random.random() < self.intervention_success_rate:
            for signs in sign_dict:
                if signs == "COVERED_SKIN_TEMPERATURE":
                    if temperature_penalty(sign_dict[signs]) < 0:
                        delta = self.np_random.normal(1.5, 0.5)
                        sign_dict[signs] -= delta
                elif signs == "PULSE_RATE":
                    if pulse_penalty(sign_dict[signs]) < 0:
                        delta = self.np_random.normal(15, 5)
                        sign_dict[signs] -= delta
                elif signs == "RESPIRATORY_RATE":
                    if respiratory_penalty(sign_dict[signs]) < 0:
                        delta = self.np_random.normal(10, 10 / 3)
                        sign_dict[signs] -= delta
                elif signs == "SPO2":
                    if spo2_penalty(sign_dict[signs]) < 0:
                        delta = self.np_random.normal(5, 5 / 3)
                        sign_dict[signs] += delta

        if min_max:
            # renormalize
            sign_dict = {
                sign: (c - min_max[sign][0]) / (min_max[sign][1] - min_max[sign][0])
                for sign, c in sign_dict.items()
            }

        return sign_dict


class VitalSignsSimpleLang(LanguageWrapper):
    def __init__(self, path: str, **kwargs):
        env = VitalSignsSimple(path, **kwargs)
        super().__init__(env)

    _state_mapping = {
        "PULSE_RATE": "Pulse rate",
        "RESPIRATORY_RATE": "Respiratory rate",
        "SPO2": "SPO2",
        "COVERED_SKIN_TEMPERATURE": "Covered skin temperature",
    }

    @property
    def task_text(self) -> str:
        return (
            # "You are an agent assisting doctors from a hospital to allocate wearable"
            # " devices to monitor and improve patients' vital signs."
            # " The vital signs considered include pulse rate, respiratory rate, SPO2,"
            # " and covered skin temperature.\n\n"
            # " Each device can be allocated to a patient to help manage their"
            # " vital signs. It is known that patients wearing the device can improve"
            # " their vital signs when outside the normal range, and prevent abnormality.\n\n"
            # "The normal vital signs range is defined as follows: A heart rate above 120,"
            # " a temperature above 38°C, a respiratory rate above 30, and an SPO2 rate below 90.\n\n"
            # "The reward function (negative of cost) of the decision problem is calculated as "
            # "follows: For a heart rate h, the penalty is -exp(|h-120|/17). For a temperature t,"
            # " the penalty is -exp(|t-38.0|/2). For a respiratory rate r, the penalty is -exp(|r-30|/5)."
            # " For an SPO2 rate s, the penalty is -exp(|s-90|/4).\n\n"
            # "The abnormal vital signs of patients wearing a device are reduced towards the normal range."
            # " The effect of the intervention is probabilistic, with the magnitude of the adjustment"
            # " varying for each vital sign.\n\n"
            # "### Problem description\n\n"
            # "At each timestep, you will be asked which device to allocate to the new incoming patient."
            # " Since there are only a limited number of devices, when a device is not free, you will need device "
            # " currently in use will be reallocated to the new incoming patient."
            # " You will be given the list of devices and information about whether it is currently assigned to"
            # " a patient, along with information about the vital signs of the patient.\n\n"
            # "New patients **always** need to be assigned a device."
            # " The cost function will continue to be calculated for the previous holder until they leave the"
            # f" system. A patient can wear a device for a maximum of {self.env.system_duration} timesteps, and then they"
            # " exit the system.\n\n"
            # "### Goal\n\n"
            # "The goal is to minimize the long-term cumulative cost of abnormal vital signs by"
            # " prioritizing assigning free devices to the incoming patients. Or, when no devices are free, prioritizing "
            # " taking away the device from a patient who has the least necessity for it, e.g., due to normal signs and low risk."
            "You are assisting doctors from a hospital in making optimized"
            " decisions about which patient should receive a vital sign monitor device."
            " It is known that wearing the device can help improve the patient's vital signs. "
            " and prevent abnormality."
            " Your goal is to ensure the optimal allocation of devices, such that patients with a higher"
            " risk continue wearing a device until their vital signs are within the normal range. Since"
            " there is a limited number of devices, you will need to decide which patients should stop"
            " wearing the device to reallocate it to the incoming patients. Incoming patients must"
            " always receive a device.\n"
            # "Normal Vital Sign Range: To define the normal range, we primarily follow the thresholds used for alerts signaling abnormal"
            # " vital sings in the study on vital sign monitoring devices for maternal health in Mbarara (Boatin et al. 2021) featured earlier.\n"
            "Cost Function: A cost will be inccoured if the heart rate exceeds 120, the temperature exceeds 38C, the respiratory rate exceeds 30,"
            " or if the SPO2 rate falls below 90. The cost is calculated as an exponential function of the deviation from these thresholds.\n"
            "Effect of Intervention: The abnormal vital signs of patients wearing a device are reduced towards their normal range with an estimated"
            " 70% success rate. "
        )

    @property
    def action_space_text(self) -> str:
        # return (
        #     "Choose the id of the device that will be reallocate to the new incoming patient."
        #     f"Your answer should be a single integer i from 0 to {self.env.budget - 1} (the number of devices) such that:\n"
        #     "- If device i is currently worn by a patient, then this patient will stop benefiting from the intervention.\n"
        #     "- If device i is free, then no active patient will stop benefiting from the intervention."
        #     "Your answer should start with the device id as an integer value and do not provide any additional information."
        # )
        return (
            "Choose the id of the device that will be reallocated to the new incoming patient."
            f"Your answer should be a single integer i from 0 to {self.env.budget} (the number of devices) such that:\n\n"
            "- Always choose a free device if available\n"
            "- If no free device is available, then choose device i whose current patient is at least risk or"
            " would benefit less from wearing the device."
            " Format your answer as a JSON as in the following examples: {'device': 0}, {'device': 3}"
        )

    def state_descriptor(self, *_, **__) -> str:
        env = self.env
        min_max = env.min_max
        win = env.variability_window

        lower = np.array([min_max[v][0] for v in env.vital_signs])
        upper = np.array([min_max[v][1] for v in env.vital_signs])

        desc = f"Number of devices: {env.budget}\n"

        is_free = [device["time_worn"] == 0 for device in env.device_states]
        num_free = sum(is_free)
        if num_free == 0:
            desc += "Number of free devices: none\n\n\n"
        else:
            desc += f"Number of free devices: {num_free}\n"
            free = [i for i, free in enumerate(is_free) if free]
            desc += f"Ids of free devices: {free}\n\n\n"

        desc_bits = []

        for i, device in enumerate(env.device_states):
            time_worn = device["time_worn"]

            s = f"### Device {i}\n\n"

            if is_free[i]:
                # device is free
                s += "Device is currently free.\n\n"
            else:
                # device is currently assigned
                s += "Device is currently assigned to a patient with the following description:\n\n"

                # unnormalize the values
                signs_history = np.array(device["signs_history"])
                signs_history *= (upper - lower)[:, None]
                signs_history += lower[:, None]
                mean = signs_history.mean(axis=1)
                std = signs_history.std(axis=1)

                s += f"*Timesteps wearing the device*: {time_worn}\n\n"
                for j, v in enumerate(env.vital_signs):
                    hist = ", ".join(f"{x:.2f}" for x in signs_history[j])
                    s += f"*{self._state_mapping[v]}*\n"
                    # s += f"- Recent history (oldest to newest): {hist}\n"
                    s += f"- Last value: {signs_history[j][-1]:.2f}\n"
                    s += f"- Mean: {mean[j]:.2f}\n"
                    s += f"- Standard deviation/volatility: {std[j]:.2f}\n\n"

            desc_bits.append(s)

        desc += "\n".join(desc_bits)

        return desc

    @property
    def example_rules(self) -> List[str]:
        rule_1 = (
            '{"background": "There is no advantage in having unused wearable devices",'
            ' "rule": "If there are free devices, always choose them over non-free devices",'
            ' "state relevance": "Currently, devices 0, 1 and 3 are free",'
            ' "goal relevance": "The goal is to maximize the number of patients wearing a device"}'
        )

        rule_2 = (
            '{"background": "Patients with high volatility in their vital signs are at higher risk of abnormal vital signs even if their last observed signs are normal",'
            ' "rule": "Prioritize reallocating the devices of patients with low volatility in their vital signs if there are no free devices."n'
            ' "state relevance": "Currently, no device is free. All patients have normal signs; however, patient wearing device #3 has a high volatility in its blood pressure (128 +- 30)",'
            ' "goal relevance": "Reallocating the device of a patient with low volatility is safer. The agent\'s goal is to ensure patients at risk are wearing a device"}'
        )

        rule_3 = (
            '{"background": "Patients with abnormal vital signs will benefit from continued use of the device",'
            ' "rule": "Prioritize reallocating the devices of patients with normal vital signs over those with abnormal vital signs",'
            ' "state relevance": "In the current problem state, the patient wearing device #2 has low SPO2 (85%), while the vital signs of other patients are normal",'
            ' "goal relevance": "The agent goal is to maximize the benefits to wear the device to revert abnormal vital signs to normal"}'
        )
        return [rule_1, rule_2, rule_3]


if __name__ == "__main__":
    # env = VitalSignsLang(path="models/uganda.npz", parse_action=True)

    # # task step
    # print(f"\n\n== Task Step ==\n{env.task_text}")

    # # action space text
    # print(f"\n\n== Action Space Text ==\n{env.action_space_text}")

    # # reset
    # obs, info = env.reset()
    # print(f"Initial state:\n {obs[1]}")

    # print(f"\n\n== Step: {0} == ")
    # ## One step
    # obs, reward, terminated, truncated, info = env.step("[]")
    # print(f"State:\n {obs[1]}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")

    # print(f"\n\n== Step: {1} == ")
    # ## Four steps
    # obs, reward, terminated, truncated, info = env.step("[2, 3]")
    # print(f"State:\n {obs[1]}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")

    # print(f"\n\n== Step: {2} == ")
    # ## One step
    # obs, reward, terminated, truncated, info = env.step("[]")
    # print(f"State:\n {obs[1]}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")

    # print(f"\n\n== Step: {3} == ")
    # ## Four steps
    # obs, reward, terminated, truncated, info = env.step("[4, 5]")
    # print(f"State:\n {obs[1]}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")

    # print(f"\n\n== Step: {4} == ")
    # ## One step
    # obs, reward, terminated, truncated, info = env.step("[]")
    # print(f"State:\n {obs[1]}")
    # print(f"Reward: {reward}")
    # print(f"Terminated: {terminated}")
    # print(f"Truncated: {truncated}")

    # for step in range(5):
    #     print(f"\n\n== Step: {step} == ")
    #     # action
    #     obs, reward, terminated, truncated, info = env.step("[0, 1]")
    #     print(f"State:\n {obs[1]}")
    #     print(f"Reward: {reward}")
    #     print(f"Terminated: {terminated}")
    #     print(f"Truncated: {truncated}")

    import envs as E
    import gymnasium as gym

    # env = VitalSignsSimpleLang(path="models/uganda.npz")
    env = gym.make("Uganda")

    # reset
    obs, _ = env.reset()

    print(f"Initial state:\n {obs}")

    for i in range(18):
        print(f"\n\n== Step: {i} == ")
        obs, reward, terminated, truncated, _ = env.step(0)
        print(obs[1])
        print(f"Reward: {reward}")

    0
