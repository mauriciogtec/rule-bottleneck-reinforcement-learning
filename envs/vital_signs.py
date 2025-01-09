import math
import re
from typing import Optional, Sequence

import numpy as np
import pandas as pd
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
    if 90 <= spo2:
        return 0
    else:
        return -math.exp(abs(spo2 - 90) / 4)  # Exponential penalty


def blood_penalty(blood_pressure):
    if blood_pressure <= 127:
        return 0
    else:
        return -math.exp(abs(blood_pressure - 127) / 5)  # Exponential penalty


def reward_function(sign_dict, min_max, clip_value=5, scaler=0.1):
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

    # Clip the reward to avoid large values
    reward = np.clip(scaler * reward, -clip_value, clip_value)

    return reward


class VitalSignsEnv(Env):
    """'Class to simulate the environment
    for the online RL agent"""

    def __init__(
        self,
        path: str,
        init_agents=2,  # B=3 in the paper
        max_num_agents=10,  # N=20 in the paper
        budget=2,  # They have a budget, which does not necessarily equal to init_agent
        t_min=1,  # t_min = 3 in the paper
        t_max=5,  # t_max = 5 in the paper
        joining_number=2,  # = two patients in the paper, no letter
        system_duration=10,  # = 50 in the paper, no letter
        joining_interval=5,  ## In the paper, patients join every 5 timesteps
        degree_of_arm_noise=0.15,
        intervention_success_rate=0.7,
        variability_window=5,
        T: Optional[int] = None,
    ):
        """
        Parameters:
            path: path to the gmm and minmax data
            num_agents: number of agents in the beginning
            budget: the # of medical device available
            max_num_agents: the max # of agents that could possibly appear at the same time
            T: time horizon
            t_min: minimum number of time to wear the device
            t_max: maximum number of time to wear the device
        """
        ## If the budget is larger than the max_num_agents, then there is no
        ## scarcity and no need for decision making
        if budget > max_num_agents:
            raise ValueError("Enough device for all patients, no need to train")

        # ## According to the rule, all incoming agents should receive the device
        if init_agents > budget:
            raise ValueError("Not enough device to allocate at the beginning")

        # load GMM
        self._load_gmm(path)

        ## We are in a finite time horizon
        self.T = T
        self.remaining_planning_length = T
        ## Number of agents at the initial time step
        self.init_agents = init_agents
        self.num_agents = init_agents
        self.max_num_agents = max_num_agents
        self.budget = budget
        self.t_min = t_min
        self.t_max = t_max
        self.joining_number = joining_number
        self.system_duration = system_duration
        self.joining_interval = joining_interval

        # self.leaving_time = leaving_time
        self.next_agent_id = (
            init_agents  ## The id to the first patient arrive in the next round
        )

        # # inter arrival time
        # self.inter_arrival_steps = system_duration // joining_rate

        self.degree_of_arm_noise = degree_of_arm_noise
        self.intervention_success_rate = intervention_success_rate
        self.variability_window = variability_window

        ## Compute the max num of agents

        ## Actions is a list of index that denotes agents who are pulled
        # total_action = math.comb(max_num_agents, budget)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(max_num_agents,), dtype=int
        )

        #     The observation space is
        #     (patient agent id [bounded by maximum number of agents],
        #     vital signs + variability + sign history for each vital sign
        #     + binary flag about device allocation + time since joined)
        # '''

        # for the time being we don't need the history
        # self.obs_dim = 3 + 3 + 3 * variability_window + 2
        self.obs_dim = 3 + 3 + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(max_num_agents * self.obs_dim,),
            dtype=float,
        )

        ## Track agent states
        self.agent_states = []

        ## Random number generator
        self.rng = np.random.default_rng()

        ## Initialize agents at time step 0
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the agents' states at time step 0"""
        for agent_id in range(self.num_agents):
            ## Sample the initial state for each agent from the mixture
            ## of Gaussian models in IAAI paper
            state, component, mean, cov = self._sample_agent()
            current_vital = state[0]
            variability = state[1]
            signs_history = state[2]

            ## All incoming agents must receive the device
            # device_flag = 1  # commenting cause abusing np.append (should be used as replacement of list appent)
            # time_joined = 1
            # state_with_flag = np.append(state, device_flag)
            # overall_state = np.append(state_with_flag, time_joined)
            new_agent_info = {
                "id": agent_id,
                "mean": mean,
                "cov": cov,
                "component": component,
                "vitals": current_vital,
                "variability": variability,
                "signs_history": signs_history,
                "has_device": 1,
                "time_joined": 1,
            }
            self.agent_states.append(new_agent_info)

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

    def _read_gmm_minmax(self):
        """read the meam, cov, and min_max of the mixture model"""
        gmm = self.gmm
        min_max = self.min_max
        means = gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_

        # Normalize the weights to ensure they sum to 1
        weights /= np.sum(weights)

        # Sample an index based on the weights
        component = self.rng.choice(len(weights), p=weights)

        mean = means[component]
        cov = covariances[component]

        return mean, cov, min_max

    def _reward_function(self, sign_dict):
        reward = 0
        for signs in sign_dict:
            if signs == "COVERED_SKIN_TEMPERATURE":
                reward += temperature_penalty(sign_dict[signs])
            elif signs == "PULSE_RATE":
                reward += pulse_penalty(sign_dict[signs])
            elif signs == "RESPIRATORY_RATE":
                reward += respiratory_penalty(sign_dict[signs])
            elif signs == "SPO2":
                reward += spo2_penalty(sign_dict[signs])
        return reward

    # sign_dict, min_max, intervention_success_rate
    def _improve_vital_signs3(self, sign_dict):
        """improve_vital_signs: Another model of the positive effect of intervention
        (assigning a medical device). The medical staff reacts to the alert 70%
        of time in reality, and the abnormal vital sign is adjusted probablistically
        towards the normal. This seems to be the model used in the final paper.
        """

        mean, cov, min_max = self._read_gmm_minmax()
        if min_max:
            # normalize
            sign_dict = {
                sign: c * (min_max[sign][1] - min_max[sign][0]) + min_max[sign][0]
                for sign, c in sign_dict.items()
            }

        # print(sign_dict)
        if self.rng.random() < self.intervention_success_rate:
            for signs in sign_dict:
                if signs == "COVERED_SKIN_TEMPERATURE":
                    if temperature_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(1.5, 0.5)
                elif signs == "PULSE_RATE":
                    if pulse_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(15, 5)
                elif signs == "RESPIRATORY_RATE":
                    if respiratory_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(
                            10, 10 / 3
                        )
                elif signs == "SPO2":
                    if spo2_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] + self.rng.normal(3, 1)

        if min_max:
            # renormalize
            sign_dict = {
                sign: (c - min_max[sign][0]) / (min_max[sign][1] - min_max[sign][0])
                for sign, c in sign_dict.items()
            }

        return sign_dict

    def _conditional_sample_mnd(self, vital_values, given_indices):
        """
        Sample from the conditional distribution of a multivariate Normal Distribution
        """
        mean, cov, min_max = self._read_gmm_minmax()
        all_indices = np.arange(len(mean))
        remaining_indices = np.setdiff1d(all_indices, given_indices)

        # Convert to DataFrame
        df = pd.DataFrame(cov[0])

        # Print nicely formatted
        # print(df.to_string(index=False, float_format="%.8f"))
        # print(means,weights)
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

        v = self.rng.multivariate_normal(mean=conditional_mean, cov=conditional_cov)

        return np.clip(v, 0, 1)

    # current_values, min_max, intervention_success_rate, mean=None, cov=None,
    def _interventions(self, vital_values):
        """interventions: This function models the effect of intervention. If the patient's value
        falls in the normal range, then the patient's next state will be sampled from a multivariate
        Guassian from this current state

        If the patient's vital sign shows abnormality, then there is a 30% chance the doctors do not
        intervene, and there is a 70% chance the intervention creates a positive effect on the patient.
        After applying the positive effect, the patient's new state will be the condition for sampling
        the next state
        """
        mean, cov, min_max = self._read_gmm_minmax()
        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        if self._reward_function(dict(zip(vital_signs, vital_values))) >= 0:
            return self._conditional_sample_mnd(vital_values, given_indices)
        else:
            # new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
            # print("Old", current_values)
            new_signs = self._improve_vital_signs3(dict(zip(vital_signs, vital_values)))
            # print("NEW",[new_signs[vital] for vital in vital_signs])
            return self._conditional_sample_mnd(
                [new_signs[vital] for vital in vital_signs], given_indices
            )

    def _simulate_one_step(self, agent_index, intervention=False):
        """simulate_one_step: based on the current value, calculate what's the next state for vital signs,
        the variance of vital sign for the past five timesteps, and the reward
        """

        mean, cov, min_max = self._read_gmm_minmax()
        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        agent_info = self.agent_states[agent_index]
        vital_values = agent_info["vitals"]
        signs_history = agent_info["signs_history"]

        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        if intervention:
            next_signs = self._interventions(vital_values)
        else:
            next_signs = self._conditional_sample_mnd(vital_values, given_indices)

        for i in range(len(vital_signs)):
            del signs_history[i][0]
            signs_history[i].append(next_signs[i])

        # Note: Mauricio changed to standard deviation, better normalization
        variability = np.array([np.std(l) for l in signs_history])

        reward = self._reward_function(dict(zip(vital_signs, next_signs)))
        return [next_signs, variability, signs_history], reward

    def _resample_values(self):
        """resample_values: You sample from a multivariate Gaussian for your initial value,
        and you sample conditioned on the previous value until you have enough sign history to
        calculate variability

        Then you return the current signs, the variability of the past timesteps, the past
        vital sign values, and the corresponding reward of the currrent vital sign
        """

        mean, cov, min_max = self._read_gmm_minmax()
        variability_window = self.variability_window

        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        sample = self.rng.multivariate_normal(mean=mean, cov=cov)
        sample = np.clip(sample, 0, 1)

        current_signs = [sample[i] for i in given_indices]
        signs_history = [[] for _ in range(len(vital_signs))]
        for i in range(len(vital_signs)):
            signs_history[i].append(sample[i])

        for _ in range(variability_window - 1):
            current_signs = self._conditional_sample_mnd(current_signs, given_indices)
            for i in range(len(vital_signs)):
                signs_history[i].append(current_signs[i])

        # print(signs_history)
        # for l in signs_history:
        # print(l,np.var(l))
        variability = np.array([np.std(l) for l in signs_history])
        # print(variability)
        reward = self._reward_function(dict(zip(vital_signs, current_signs)))
        return [current_signs, variability, signs_history], reward

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
        component = self.rng.choice(len(weights), p=weights)

        means = gmm.means_
        covariances = gmm.covariances_
        mean = means[component]
        cov = covariances[component]
        state, _ = self._resample_values()

        perturb = self.rng.choice([i for i in range(len(weights)) if i != component])

        x = self.rng.uniform(0, self.degree_of_arm_noise)
        y = self.rng.uniform(0, self.degree_of_arm_noise)

        mean = (1 - x) * mean + x * means[perturb]
        cov = (1 - y) * cov + y * covariances[perturb]

        # print(mean,cov)
        # pertubation
        return state, component, mean, cov

    def _change_state_to_obs(self):
        agent_states = self.agent_states

        # Initialize an empty list to hold rows
        agent_matrix = np.zeros((self.max_num_agents, self.obs_dim))

        # Iterate over each agent's state
        for j, agent in enumerate(agent_states):
            # Extract values from the dictionary
            vitals = list(agent["vitals"])  # 1D array with 3 values
            variability = list(agent["variability"])  # List with 3 elements
            # signs_history = agent["signs_history"] # History not needed for now
            has_device = agent["has_device"]
            time_joined = agent["time_joined"] / self.system_duration

            # Concatenate all components into a single row
            agent_matrix[j, :] = np.array(
                vitals + variability + [has_device, time_joined]
            )
        return agent_matrix.flatten()

    def _advance_step(self, action, step_forward, check_device=False):
        ## This variable tracks how many devices are kept from last round
        ## Only true when there is incoming agent
        device_kept_from_last_round = 0
        if check_device:
            if not (step_forward == 1):
                raise ValueError(
                    "Need to handle incoming agents before advancing to the next step"
                )
        # else:
        #     ## When there is no incoming agent, action must be all 0's
        #     if not np.all(action == 0): # Check if all elements in the array are 0
        #         raise ValueError("Invalid Action: when there is no incoming agents, you cannot take device away from people")

        ## Accumulated reward from this round
        overall_reward_this_step = 0
        leaving_shifting_index = 0
        # leaving_agent_index = []

        ## Number of steps ahead
        for j in range(step_forward):
            for i in range(len(self.agent_states)):
                i -= leaving_shifting_index
                agent_info = self.agent_states[i]
                agent_id = agent_info["id"]
                time_joined = agent_info["time_joined"]
                has_device = agent_info["has_device"]
                mean = agent_info["mean"]
                cov = agent_info["cov"]

                if (action[i] == 0) and (has_device == 1):
                    ## The agent has device in the last round and
                    ## keeps the device

                    ## We temporarily remove the t_max constraint

                    # if time_joined >= self.t_max:
                    #     raise ValueError(
                    #         f"patient {agent_id} has received device for more than {self.t_max} rounds"
                    #     )
                    if check_device:
                        device_kept_from_last_round += 1
                    # Update the state under active action
                    new_state, reward = self._simulate_one_step(i, intervention=True)
                    overall_reward_this_step += reward
                elif (action[i] == 0) and (has_device == 0):
                    ## The agent doesn't have device in the last round
                    ## and maintain the status
                    # Update the state under passive action
                    new_state, reward = self._simulate_one_step(i, intervention=False)
                    overall_reward_this_step += reward
                elif (action[i] == 1) and (has_device == 1):
                    ## The agent has device in te last round and the
                    ## device got taken away

                    ## We temporarily remove the t_min constraint
                    # if time_joined < self.t_min:
                    #     raise ValueError(
                    #         f"Patient {agent_id} has not received device by at least {self.t_min} rounds"
                    #     )
                    # Update the state under active action
                    new_state, reward = self._simulate_one_step(i, intervention=False)
                    overall_reward_this_step += reward
                    has_device = 0  ## change the device status
                else:
                    ## The agent doesn't have device in the last round
                    ## and you want to take device from them, raise an Error
                    raise ValueError(f"No device to be taken from Patient {agent_id}")

                time_joined += 1
                new_vitals = new_state[0]
                new_variability = new_state[1]
                new_signs_history = new_state[2]
                self.agent_states[i]["vitals"] = new_vitals
                self.agent_states[i]["variability"] = new_variability
                self.agent_states[i]["signs_history"] = new_signs_history
                self.agent_states[i]["time_joined"] = time_joined
                self.agent_states[i]["has_device"] = has_device

                if time_joined >= self.system_duration:
                    # leaving_agent_index.append(i)
                    # print(f"I hit here")
                    self.agent_states.pop(i)
                    leaving_shifting_index += 1
                    self.num_agents -= 1
            # for i in leaving_agent_index:
            #     print
            #     self.agent_states.pop(i)
            self.remaining_planning_length -= 1
            if self.remaining_planning_length <= 0:
                break

        return device_kept_from_last_round, overall_reward_this_step

    def step(self, action):
        # print(f"the current action is {action}")
        # print(f"remaining length is {self.remaining_planning_length}")
        # print(self.agent_states)
        time_passed = self.T - self.remaining_planning_length
        print(f"Beginning time_passed is {time_passed}")

        if time_passed % self.joining_interval == 0:
            # print(f"Location A")
            num_joining_agents = self.joining_number
            if num_joining_agents > self.budget:
                raise ValueError("Not enough device for every incoming patient")
            step_forward = 1
            check_device = True
            device_kept_from_last_round, overall_reward_this_step = self._advance_step(
                action, step_forward, check_device
            )

            if device_kept_from_last_round + num_joining_agents > self.budget:
                raise ValueError(
                    "Not enough device to allocate according to the policy"
                )

            ## Update the number of agents
            self.num_agents = self.num_agents + num_joining_agents
            if self.num_agents > self.max_num_agents:
                raise ValueError("Exceeds the max number of agents")

            for i in range(num_joining_agents):
                state, component, mean, cov = self._sample_agent()
                agent_id = self.next_agent_id
                self.next_agent_id += 1
                new_agent_info = {
                    "id": agent_id,
                    "mean": mean,
                    "cov": cov,
                    "component": component,
                    "vitals": state[0],
                    "variability": state[1],
                    "signs_history": state[2],
                    "has_device": 1,
                    "time_joined": 1,  # TODO: why would this be the case in step? likely wrong
                }
                self.agent_states.append(new_agent_info)
        else:
            # print(f"Location B")
            num_joining_agents = 0
            step_forward = self.joining_interval - 1
            check_device = False
            device_kept_from_last_round, overall_reward_this_step = self._advance_step(
                action, step_forward, check_device
            )

        # self.remaining_planning_length -= 1
        # Check if the planning is done
        if self.remaining_planning_length <= 0:
            done = True
        else:
            done = False

        # Set place holder for truncated and info
        truncated = False
        info = {}

        obs = self._change_state_to_obs()

        return obs, overall_reward_this_step, done, truncated, info

    def render(self):
        pass

    def reset(self, seed=None, options={}):
        # Set the seed
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Reinitialize agent states
        self.remaining_planning_length = self.T
        self.num_agents = self.init_agents
        self.agent_states = []

        # Initialize agents at time step 0
        self._initialize_agents()

        self.next_agent_id = self.num_agents
        info = {}

        obs = self._change_state_to_obs()
        return obs, info


class VitalSignsLang(LanguageWrapper):
    """
    A wrapper for the VitalSigns environment.
    """

    _state_mapping = {
        "PULSE_RATE": "Pulse rate",
        "RESPIRATORY_RATE": "Respiratory rate",
        "SPO2": "SPO2",
        "COVERED_SKIN_TEMPERATURE": "Covered skin temperature",
    }

    def __init__(self, path: str, parse_action: str = False, **kwargs):
        env = VitalSignsEnv(path, **kwargs)
        super().__init__(env, parse_action=parse_action)

    @property
    def task_text(self) -> str:
        return (
            "You are assisting doctors from a hospital in making optimized"
            " decisions about which patient should receive a vital sign monitor device."
            " You will determine the device allocation by considering the patients' current"
            " vital signs and their recent variability."
            " Your goal is to optimize a reward function, where a cost (negative reward)"
            " is incurred for each patient whose vital signs are outside the normal range."
            " It is known that wearing the device can help improve the patient's vital signs. "
            " and prevent abnormality.\n"
            "Normal Vital Sign Range: To define the normal range, we primarily follow the thresholds used for alerts signaling abnormal"
            " vital sings in the study on vital sign monitoring devices for maternal health in Mbarara (Boatin et al. 2021) featured earlier:"
            " A heart rate above 120, a temperature above 38C, a respiratory rate above 30, and an SPO2 rate below 90 are considered"
            " abnormal.\n"
            "Reward Function: For a heart rate h, the penalty is-exp (|h-120|/17). For a temperature t, the penalty is -exp(|t-38.0|/2)."
            " For a respiratory rate r, the penalty is -exp(|r-30|/5). For an SPO2 rate s, the penalty is -exp(|s-90|/4).\n"
            "Effect of Intervention: The abnormal vital signs of patients wearing a device are reduced towards the normal range with an estimated"
            " 70% success rate. The effect of the intervention is probabilistic, with the magnitude of the adjustment varying for each vital sign."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "A vector which contains a subset of the indices of patients currently in"
            " the system. Each patient whose index appears in the vector will be"
            " stop wearing the device. The device will be reallocate to the new patients"
            " For example, [0, 1] means that the first two patients will stop wearing the device."
            " ### Example answers: [0, 1], [1, 2], [0, 2], [0, 1]"
        )

    def state_descriptor(self, *_, **__) -> str:
        """
        Convert the observation into a text description specific to the environment

        Returns:
            str: The text description of the observation
        """
        env = self.env
        min_max = self.env.min_max  # ranges for unnormalized values

        agents_with_device = [d for d in env.agent_states if d["has_device"]]
        time_passed = self.env.T - self.env.remaining_planning_length
        print(f"After time passed is {time_passed}")
        if time_passed % self.env.joining_interval == 0:
            num_joining_agents = self.env.joining_number
        else:
            num_joining_agents = 0

        desc = (
            f"Number of agents: {env.num_agents}\n"
            f"Number of incoming agents: {num_joining_agents}\n"
            f"Number of available devices for the new patients: {len(agents_with_device)}\n"
            f"Agents with available device: {', '.join(str(d['id']) for d in agents_with_device)}\n"  ## TODO
            "Vital signs of those agents:"
        )

        for i, d in enumerate(env.agent_states):
            # only list agents with device
            has_device = d["has_device"]
            if not has_device:
                continue

            time_joined = d["time_joined"]

            aux = []
            for j, v in enumerate(env.vital_signs):
                key = self._state_mapping[v]
                sign_value, sign_variability = d["vitals"][j], d["variability"][j]
                # TODO # use min_max to unnormalize
                # something like this...
                sign_value = (
                    sign_value * (min_max[v][1] - min_max[v][0]) + min_max[v][0]
                )
                sign_variability = (
                    sign_variability * (min_max[v][1] - min_max[v][0]) + min_max[v][0]
                )
                val_text = f"{sign_value:.2f} +/- {sign_variability:.2f}"
                aux.append((key, val_text))

            key = "Time steps since joined (will exit the system after 50 steps)"
            aux.append((key, time_joined))

            desc += f"\n=== Agent {i} ===\n" + "\n".join(f"{k}: {v}" for k, v in aux)

            if has_device:
                desc += (
                    "\nHas a device available to reallocate to the incoming patients."
                )

        return desc

    # def action_parser(self, action_string: str) -> Sequence[int]:
    #     """This functions takes a list of indices in text mode (with possible spaces and unnecessary text)
    #     and returns a binary list of indices"""
    #     # 1. extract all integers
    #     int_list = [int(s) for s in re.findall(r"\d+", action_string)]

    #     if not int_list:  # If int_list is empty
    #         raise ValueError("Devices have not been assigned")

    #     # 2. covert to binary list with size max_num_agents
    #     action = np.zeros(self.env.max_num_agents, dtype=int)
    #     action[np.array(int_list)] = 1

    #     return action
    def action_parser(self, action_string: str) -> Sequence[int]:
        """This function takes a list of indices in text mode (with possible spaces and unnecessary text)
        and returns a binary list of indices representing whether to take a device away from each agent.
        """
        # 1. Extract all integers
        int_list = [int(s) for s in re.findall(r"\d+", action_string)]

        # Initialize the binary action list
        action = np.zeros(self.env.max_num_agents, dtype=int)

        # 2. Loop through agent states and determine the action for each agent
        for i, agent_state in enumerate(self.env.agent_states):
            has_device = agent_state[
                "has_device"
            ]  # Check if the agent currently has a device

            if has_device == 1:
                if i in int_list:
                    # If the agent has a device and is in the list, no action needed (keep device)
                    action[i] = 0
                else:
                    # If the agent has a device but is NOT in the list, take the device away
                    action[i] = 1
            else:
                # If the agent does not have a device, no action needed
                action[i] = 0

        return action


class VitalSignsSimple(Env):
    def __init__(
        self,
        path: str,
        init_agents=1,  # B=3 in the paper
        # max_num_agents=10,  # N=20 in the paper
        budget=2,  # They have a budget, which does not necessarily equal to init_agent
        # t_min=1,  # t_min = 3 in the paper
        # t_max=5,  # t_max = 5 in the paper
        system_duration=10,  # = 50 in the paper, no letter
        degree_of_arm_noise=0.15,
        intervention_success_rate=0.7,
        variability_window=5,
        # joining_number=2,  # Here, vital signs only advance after N patients join
        joining_interval=5,  # Here, simulate the number of internal vital signs steps
        T: Optional[int] = None,  # planning length / for finite horizon evaluation, only use for evaluation
    ):
        """
        Parameters:
            path: path to the gmm and minmax data
            num_agents: number of agents in the beginning
            budget: the # of medical device available
            max_num_agents: the max # of agents that could possibly appear at the same time
            T: time horizon
            t_min: minimum number of time to wear the device
            t_max: maximum number of time to wear the device
        """
        ## Check inputs
        assert (
            init_agents <= budget
        ), "The number of agents should be less than or equal to the budget"

        ## Random number generator
        self.rng = np.random.default_rng()
        self.T = T

        # load GMM
        self._load_gmm(path)

        ## We are in a finite time horizon
        # self.T = T
        # self.remaining_planning_length = T
        ## Number of agents at the initial time step
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
        self.observation_space = spaces.Box(0, 1.0, shape=(self.total_dim,))

        ## Track agent states
        self.device_states = [{} for _ in range(self.budget)]

    def _initialize_agents(self, init_agents=None):
        """Initialize the agents' states at time step 0"""
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

        return agent_matrix.flatten()

    def reset(self, seed=None, options=None):
        # Reset time counter
        self.t = 0

        # Set seed
        self.rng = np.random.default_rng(seed)

        # Empty the device states

        # Initialize agents at time step 0
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

        if not is_free:
            # simulate the rest of the time for the current holder
            mean, cov = self.device_states[action]["gmm"]
            time_worn = self.device_states[action]["time_worn"]

            if self.T is not None:
                remaining = min(self.T - self.t, self.system_duration - time_worn)
            else:
                remaining = self.system_duration - time_worn

            for _ in range(remaining):
                # device is free, simulate remaining time without device
                state, r = self._simulate_one_step(action, intervention=False)

                # update agent state
                self.device_states[action]["time_worn"] += 1
                self.device_states[action]["vitals"] = current_vital
                self.device_states[action]["variability"] = variability
                self.device_states[action]["signs_history"] = signs_history

                reward += r

        # Assign the device to the new patient
        self.device_states[action]["time_worn"] = 1
        self.device_states[action]["vitals"] = current_vital
        self.device_states[action]["variability"] = variability
        self.device_states[action]["signs_history"] = signs_history
        self.device_states[action]["gmm"] = mean, cov

        # == 2. Update the remaining device/agents ==
        for i in range(self.budget):
            is_free = self.device_states[i]["time_worn"] == 0
            if not is_free and i != action:
                self.device_states[i]["time_worn"] += 1

                # advance the vital signs
                state, r = self._simulate_one_step(i, intervention=True)

                # update agent state
                self.device_states[i]["vitals"] = state[0]
                self.device_states[i]["variability"] = state[1]
                self.device_states[i]["signs_history"] = state[2]

                reward += r

                # remove them from the system if system duration reached
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
                    for t in range(time_worn, self.system_duration):
                        # device is free, simulate remaining time without device
                        state, r = self._simulate_one_step(i, intervention=True)

                        # update agent state
                        self.device_states[i]["time_worn"] = t + 1
                        self.device_states[i]["vitals"] = state[0]
                        self.device_states[i]["variability"] = state[1]
                        self.device_states[i]["signs_history"] = state[2]

                        reward += r

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

        v = self.rng.multivariate_normal(mean=conditional_mean, cov=conditional_cov)

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
        component = self.rng.choice(len(weights), p=weights)

        means = gmm.means_
        covariances = gmm.covariances_
        mean = means[component]
        cov = covariances[component]
        state, _ = self._resample_values(mean, cov)

        perturb = self.rng.choice([i for i in range(len(weights)) if i != component])

        x = self.rng.uniform(0, self.degree_of_arm_noise)
        y = self.rng.uniform(0, self.degree_of_arm_noise)

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

        sample = self.rng.multivariate_normal(mean=mean, cov=cov)
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
        if self.rng.random() < self.intervention_success_rate:
            for signs in sign_dict:
                if signs == "COVERED_SKIN_TEMPERATURE":
                    if temperature_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(1.5, 0.5)
                elif signs == "PULSE_RATE":
                    if pulse_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(15, 5)
                elif signs == "RESPIRATORY_RATE":
                    if respiratory_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] - self.rng.normal(
                            10, 10 / 3
                        )
                elif signs == "SPO2":
                    if spo2_penalty(sign_dict[signs]) < 0:
                        sign_dict[signs] = sign_dict[signs] + self.rng.normal(3, 1)

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
            "You are an agent assisting doctors from a hospital to allocate wearable"
            " devices to monitor and improve patients' vital signs."
            " The vital signs considered include pulse rate, respiratory rate, SPO2,"
            " and covered skin temperature.\n\n"
            " Each device can be allocated to a patient to help manage their"
            " vital signs. It is known that patients wearing the device can improve"
            " their vital signs when abormal, and prevent abnormality.\n\n"
            "The normal vital signs range is defined as follows: A heart rate above 120,"
            " a temperature above 38°C, a respiratory rate above 30, and an SPO2 rate below 90.\n\n"
            "The reward function (negative of cost) of the decision problem is calculated as "
            "follows: For a heart rate h, the penalty is -exp(|h-120|/17). For a temperature t,"
            " the penalty is -exp(|t-38.0|/2). For a respiratory rate r, the penalty is -exp(|r-30|/5)."
            " For an SPO2 rate s, the penalty is -exp(|s-90|/4).\n\n"
            "The abnormal vital signs of patients wearing a device are reduced towards the normal range."
            " The effect of the intervention is probabilistic, with the magnitude of the adjustment"
            " varying for each vital sign.\n\n"
            "### Problem description\n\n"
            "At each time step, you will be asked which device to allocate to the new incoming patient."
            " Since there are only a limited number of devices, when a device is not gree, you will need device "
            " currently in use will be reallocated to the new incoming patient."
            " You will be given the list of devices and information about whether it is currently assigned to"
            " a patient, along with information about the vital signs of the patient.\n\n"
            "New patients **always** need to be assigned a device."
            " The cost function will continue to be calculated for the previous holder until they leave the"
            f" system. A patient can wear a device for a maximum of {self.env.system_duration} time steps, and then they"
            " exit the system.\n\n"
            "### Goal\n\n"
            "The goal is to minimize the long-term cumulative cost of abnormal vital signs by intelligently"
            " choosing which device to allocate to the new patient, considering that the previous holder"
            " will stop benefiting from the device when they stop wearing it."
        )

    @property
    def action_space_text(self) -> str:
        return (
            "\n\n### Expected Answer\n\nA single integer with the device id, which goes from zero to the number of devices."
            f" The answer should be a single integer from [0, 1, ..., {self.env.budget - 1}].\n\n"
            "Even if a device is in use, you can still assign it to the new patient. Assining free devices if available,"
            " otherwise, reassign a device currently in use."
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

                s += f"**Patient information**\n\n"
                s += f"*Time steps wearing the device*\n{time_worn}\n\n"
                for j, v in enumerate(env.vital_signs):
                    hist = ", ".join(f"{x:.2f}" for x in signs_history[j])
                    s += f"*{self._state_mapping[v]}*\n"
                    s += f"- Recent history (oldest to newest): {hist}\n"
                    s += f"- Last value: {signs_history[j][-1]:.2f}\n"
                    s += f"- Mean: {mean[j]:.2f}\n"
                    s += f"- Standard deviation: {std[j]:.2f}\n\n"

            desc_bits.append(s)

        desc += "\n".join(desc_bits)

        return desc


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
