## Create gym environment here.
# import os
import numpy as np

# import torch
from gymnasium import Env, spaces

# from gymnasium.spaces import Discrete, Box
from sklearn.mixture import GaussianMixture
import math
import random
import pandas as pd


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


def conditional_sample_mnd(given_values, given_indices, mean, cov):
    """
    Sample from the conditional distribution of a multivariate Normal Distribution

    Parameters:
    - gmm: Fitted GaussianMixture object
    - given_values: The values of the given variables
    - given_indices: The indices of the given variables

    Returns:
    - Sample from the conditional distribution
    """
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
    conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (
        given_values - mean_given
    )
    conditional_cov = (
        cov_remaining_remaining
        - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining
    )

    return np.clip(
        np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov), 0, 1
    )


# def clean_data(vital_signs,p_df,min_max):  # TODO: this function is not needed clearly, thin wrapper
#     for sign in vital_signs:
#         p_df[sign] = reverse_min_max_normalize(p_df[sign], min_max[sign][0], min_max[sign][1])
#     return p_df


# def reverse_min_max_normalize(column, min_val, max_val):
#     return column * (max_val - min_val) + min_val


def reward_function(sign_dict, rev_norm=False, o_values=None):
    # if rev_norm:
    #     # print(sign_dict)
    #     sign_dict = clean_data(vital_signs, sign_dict, o_values)

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


def improve_vital_signs3(
    sign_dict, min_max, intervention_success_rate, rev_norm=False, o_values=None
):
    """improve_vital_signs: Another model of the positive effect of intervention
    (assigning a medical device). The medical staff reacts to the alert 70%
    of time in reality, and the abnormal vital sign is adjusted probablistically
    towards the normal. This seems to be the model used in the final paper.

    Input:
        - sign_dict: A dictionary whose key is the vital sign name, and value
            is the corresponding value
        - rev_norm: a binary variable indicating whether you need to clean data
            in the beginning

    Output:
        - sign_dict: A dictionary with vital signs as key and modified value as
            values
    """
    # if rev_norm:
    #     # print(sign_dict)
    #     sign_dict = clean_data(vital_signs, sign_dict, o_values)

    if min_max:
        # normalize
        sign_dict = {
            sign: c * (min_max[sign][1] - min_max[sign][0]) + min_max[sign][0]
            for sign, c in sign_dict.items()
        }

    # print(sign_dict)
    if random.random() < intervention_success_rate:
        for signs in sign_dict:
            if signs == "COVERED_SKIN_TEMPERATURE":
                if temperature_penalty(sign_dict[signs]) < 0:
                    sign_dict[signs] = sign_dict[signs] - np.random.normal(1.5, 0.5)
            elif signs == "PULSE_RATE":
                if pulse_penalty(sign_dict[signs]) < 0:
                    sign_dict[signs] = sign_dict[signs] - np.random.normal(15, 5)
            elif signs == "RESPIRATORY_RATE":
                if respiratory_penalty(sign_dict[signs]) < 0:
                    sign_dict[signs] = sign_dict[signs] - np.random.normal(10, 10 / 3)
            elif signs == "SPO2":
                if spo2_penalty(sign_dict[signs]) < 0:
                    sign_dict[signs] = sign_dict[signs] + np.random.normal(3, 1)

    if min_max:
        # renormalize
        sign_dict = {
            sign: (c - min_max[sign][0]) / (min_max[sign][1] - min_max[sign][0])
            for sign, c in sign_dict.items()
        }

    return sign_dict


def interventions(
    current_values,
    min_max,
    mean=None,
    cov=None,
):
    """interventions: This function models the effect of intervention. if the patient's value
    falls in the normal range, then the patient's next state will be sampled from a multivariate
    Guassian from this current state

    If the patient's vital sign shows abnormality, then there is a 30% chance the doctors do not
    intervene, and there is a 70% chance the intervention creates a positive effect on the patient.
    After applying the positive effect, the patient's new state will be the condition for sampling
    the next state

    Input:
    - current_values: current values of the vital signs
    - given_indices: what indices are given in the conditional sampling

    Output:
    - value of vital signs for the next state
    """
    vital_signs = list(min_max.keys())
    given_indices = np.arange(len(vital_signs))

    if (
        reward_function(
            dict(zip(vital_signs, current_values)), rev_norm=True, o_values=min_max
        )
        >= 0
    ):
        return conditional_sample_mnd(current_values, given_indices, mean=mean, cov=cov)
    else:
        # new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
        # print("Old", current_values)
        new_signs = improve_vital_signs3(
            dict(zip(vital_signs, current_values)), rev_norm=True, o_values=min_max
        )
        # print("NEW",[new_signs[vital] for vital in vital_signs])
        return conditional_sample_mnd(
            [new_signs[vital] for vital in vital_signs],
            given_indices,
            mean=mean,
            cov=cov,
        )
        # return resample_values(gmm,min_max,component_index=component_index)[0]


def simulate_one_step(
    current_state,
    min_max,
    intervention=False,
    mean=None,
    cov=None,
):
    """simulate_one_step: based on the current value, calculate what's the next state for vital signs,
    the variance of vital sign for the past five timesteps, and the reward

    Input:
    - current_state: current_state[0] stores the current vital sign, current_state[2] stores the
        vital signs for the past five timesteps

    Output:
    - next_signs: the vital sign for the next timestep
    - variablity: the variance of vital signs from the past five states
    - signs_history: the vital sign history for the past five states
    - reward: the reward for the next signs
    """
    current_signs = current_state[0]
    signs_history = current_state[2]
    # print(current_signs)

    vital_signs = list(min_max.keys())
    given_indices = np.arange(len(vital_signs))

    if intervention:
        next_signs = interventions(
            current_values=current_signs, min_max=min_max, mean=mean, cov=cov
        )
    else:
        next_signs = conditional_sample_mnd(
            current_signs, given_indices, mean=mean, cov=cov
        )

    for i in range(len(vital_signs)):
        del signs_history[i][0]
        signs_history[i].append(next_signs[i])

    variability = [np.var(l) for l in signs_history]

    reward = reward_function(
        dict(zip(vital_signs, next_signs)), rev_norm=True, o_values=min_max
    )
    return [next_signs, variability, signs_history], reward


def resample_values(min_max, mean, cov, variability_window):
    """resample_values: You sample from a multivariate Gaussian for your initial value,
    and you sample conditioned on the previous value until you have enough sign history to
    calculate variability

    Then you return the current signs, the variability of the past timesteps, the past
    vital sign values, and the corresponding reward of the currrent vital sign
    """

    vital_signs = list(min_max.keys())
    given_indices = np.arange(len(vital_signs))

    sample = np.clip(np.random.multivariate_normal(mean=mean, cov=cov), 0, 1)
    current_signs = [sample[i] for i in given_indices]
    signs_history = [[] for _ in range(len(vital_signs))]
    for i in range(len(vital_signs)):
        signs_history[i].append(sample[i])

    for _ in range(variability_window - 1):
        current_signs = conditional_sample_mnd(
            current_signs, given_indices, mean=mean, cov=cov
        )
        for i in range(len(vital_signs)):
            signs_history[i].append(current_signs[i])

    # print(signs_history)
    # for l in signs_history:
    # print(l,np.var(l))
    variability = [np.var(l) for l in signs_history]
    # print(variability)
    reward = reward_function(
        dict(zip(vital_signs, current_signs)), rev_norm=True, o_values=min_max
    )
    return [current_signs, variability, signs_history], reward


# ## Reconstruct the gmm and minmax object
# data = np.load('models/uganda.npz')
# means = np.array(data['means'])
# covariances = np.array(data['covariances'])
# weights = np.array(data['weights'])


# scaler_min = np.array([30.7737552, 6.41025641, 71.])
# scaler_max = np.array([234.1586382, 53.96608722, 100.])
# names = np.array(['PULSE_RATE', 'RESPIRATORY_RATE', 'SPO2'])
# # Close the data file
# data.close()
# # Reconstruct the GMM
# gmm = GaussianMixture(n_components=len(weights), covariance_type='full')

# # Manually set the parameters
# gmm.weights_ = weights
# gmm.means_ = means
# gmm.covariances_ = covariances

# # Compute the precisions_cholesky_ required by the GaussianMixture object
# gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

# # Create the min_max dictionary
# min_max = {name: [min_val, max_val] for name, min_val, max_val in zip(names, scaler_min, scaler_max)}

# vital_signs = ['PULSE_RATE',  'RESPIRATORY_RATE','SPO2']
# given_indices=[j for j in range(len(vital_signs))]


class VitalSignEnv(Env):
    """'Class to simulate the environment
    for the online RL agent"""

    def __init__(
        self,
        path: str,
        init_agents=2,  # B=3 in the paper
        max_num_agents=10,  # N=20 in the paper
        # budget = 1, # Not used in the paper
        T=20,  # T = 100 in the paper
        t_min=1,  # t_min = 3 in the paper
        t_max=5,  # t_max = 5 in the paper
        joining_rate=2,  # = two patients in the paper, no letter
        system_duration=10,  # = 50 in the paper, no letter
        # leaving_time = 5,
        degree_of_arm_noise=0.15,
        intervention_success_rate=0.7,
        variability_window=5,
    ):
        """
        Parameters:
            path: path to the gmm and minmax data
            num_agents: numb    er of agents in the beginning
            budget: the # of medical device available
            max_num_agents: the max # of agents that could possibly appear at the same time
            T: time horizon
            t_min: minimum number of time to wear the device
            t_max: maximum number of time to wear the device
            joining_rate: number of people joining the system at each time step
            leaving_time: the rate of people who leaves (number of steps to stay in the system)
        """
        ## If the budget is larger than the max_num_agents, then there is no
        ## scarcity and no need for decision making
        # assert max_num_agents > budget, "Enough device for all patients, no need to train"

        # ## According to the rule, all incoming agents should receive the device
        # assert budget > init_agents, "Not enough device to allocate at the beginning"

        # load GMM
        self._load_gmm(path)

        ## We are in a finite time horizon
        self.remaining_planning_length = T
        ## Number of agents at the initial time step
        self.init_agents = init_agents
        self.num_agents = init_agents
        self.max_num_agents = max_num_agents
        # self.budget = budget
        self.t_min = t_min
        self.t_max = t_max
        self.joining_rate = joining_rate
        # self.leaving_time = leaving_time
        self.next_agent_id = (
            init_agents  ## The id to the first patient arrive in the next round
        )

        # inter arrival time
        self.inter_arrival_steps = system_duration // joining_rate
        self.degree_of_arm_noise = degree_of_arm_noise
        self.intervention_success_rate = intervention_success_rate
        self.variability_window = variability_window

        ## Compute the max num of agents

        ## Actions is a list of index that denotes agents who are pulled
        # total_action = math.comb(max_num_agents, budget)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(max_num_agents,), dtype=int
        )

        # ''' # comments should not be strings, should start with #
        #     The observation space is
        #     (patient agent id [bounded by maximum number of agents],
        #     # of vital signs plus variance + binary flag about device allocation + time since joined)
        # '''
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_num_agents, 8), dtype=float
        )

        ## Track agent states
        self.agent_states = []

        ## Initialize agents at time step 0
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize the agents' states at time step 0"""
        for agent_id in range(self.num_agents):
            ## Sample the initial state for each agent from the mixture
            ## of Gaussian models in IAAI paper
            state, component, mean, cov = self._sample_agent()

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
                "time_joined": 1,
                "has_device": 1,
                "vitals": state,
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

    def _sample_agent(self):  # moves this to in class, all functions should be in class
        """sample_agent: you choose a component basesd on weight of each component for the multivariate
        Gaussian, then you get the sample from it.
        You perturb the vital sign mean and cov by choosing a mean and covariance from another component
        in the mixture model, and randomly sampling a influence factor to determine the magnitude of
        pertubation
        """
        gmm = self.gmm
        min_max = self.min_max

        weights = gmm.weights_

        # Normalize the weights to ensure they sum to 1
        weights /= np.sum(weights)

        # Sample an index based on the weights
        component = np.random.choice(len(weights), p=weights)

        means = gmm.means_
        covariances = gmm.covariances_
        mean = means[component]
        cov = covariances[component]
        state, _ = resample_values(
            min_max, mean=mean, cov=cov, variability_window=self.variability_window
        )

        perturb = random.choice([i for i in range(len(weights)) if i != component])

        x = random.uniform(0, self.degree_of_arm_noise)
        y = random.uniform(0, self.degree_of_arm_noise)

        mean = (1 - x) * mean + x * means[perturb]
        cov = (1 - y) * cov + y * covariances[perturb]

        # print(mean,cov)
        # pertubation
        return state, component, mean, cov


    def step(self, action):
        """
        Parameters:
            action: a list of the index of arms that's going to be pulled
        """

        num_joining_agents = np.random.poisson(lam=self.joining_rate, size=1)

        ## All new patients should receive a device
        assert (
            self.budget >= num_joining_agents
        ), "Not enough device for every incoming patient"
        ## Budget constraint cannot be violated
        assert self.budget >= len(
            action
        ), "Not enough device to allocate according to the policy"

        ## Reduce planning horizon by 1
        self.remaining_planning_length -= 1
        overall_reward_this_step = 0

        """
            Question left for Mauricio, This part is a bit confusing to me. The action is directly passed in to 
            the step function, then we determine how many agents join. Then it's unclear to me how can we guarantee 
            that all incoming patients will receive a device. 
        """

        for i in range(len(self.agent_states)):
            agent_info = self.agent_states[i]
            agent_id = agent_info["id"]
            overall_state = agent_info["state"]
            mean = agent_info["mean"]
            cov = agent_info["cov"]

            if i in action:
                if (overall_state[-2] == 1) & (overall_state[-1] >= self.t_max):
                    print(
                        f"Warning, patient {agent_id} has received device for more than {self.t_max} rounds"
                    )
                elif overall_state[-2] == 0:
                    print(
                        f"Warning: You are assigning patient {agent_id} for more than one time"
                    )
                overall_state[-2] = 1  # assign device

                ## Update the state under active action
                past_state = overall_state[:6]
                new_state, reward = simulate_one_step(
                    past_state, self.min_max, intervention=True, mean=mean, cov=cov
                )
                overall_state[:6] = new_state
                overall_reward_this_step += reward
            else:
                if (overall_state[-2] == 1) & (overall_state[-1] < self.t_min):
                    print(
                        f"Warning, patient {agent_id} has not received device by at least {self.t_min} rounds"
                    )
                overall_state[-2] = 0
                ## Update the state under active action
                past_state = overall_state[:6]
                ## Update the state under passive action
                new_state, reward = simulate_one_step(
                    past_state, self.min_max, intervention=False, mean=mean, cov=cov
                )
                overall_state[:6] = new_state
                overall_reward_this_step += reward

            ## add 1 to the joining time
            overall_state[-1] += 1
            ## update the dictionary
            self.agent_states[i]["state"] = overall_state

        # Handle patients who leave at the end
        num_leaving_agents = np.random.poisson(lam=self.leaving_time, size=1)
        assert (
            num_leaving_agents <= self.num_agents
        ), "More leaving patients than remaining patients"

        # Randomly select patients to delete
        agents_to_delete = random.sample(self.agent_states, num_leaving_agents)

        # Remove the selected elements
        for agent in agents_to_delete:
            self.agent_states.remove(agent)

        ## Update the number of agents
        self.num_agents = self.num_agents + num_joining_agents
        assert (
            self.num_agents <= self.max_num_agents
        ), "Exceeds the max number of agents"

        for i in range(num_joining_agents):
            state, component, mean, cov = self._sample_agent(
                self.gmm, self.min_max, self.given_indices
            )
            device_flag = 1
            time_joined = 1
            state_with_flag = np.append(state, device_flag)
            overall_state = np.append(state_with_flag, time_joined)

            agent_id = self.next_agent_id
            self.next_agent_id += 1
            new_agent_info = {
                "id": agent_id,
                "mean": mean,
                "cov": cov,
                "component": component,
                "time_joined": 1,  # TODO: why would this be the case in step? likely wrong
                "has_device": 1,
                "vitals": state,
            }
            self.agent_states.append(new_agent_info)

        # Check if the planning is done
        if self.planning_length <= 0:
            done = True
        else:
            done = False

        # Set place holder for info
        info = {}

        # this is wrong, the observation should be in vector form
        # self.agent_states is a list of dictionaries and it is the internal state instead
        return self.agent_states, overall_reward_this_step, done, info

    def render(self):
        pass

    def reset(self):
        # self.remaining_planning_length = T

        # Reinitialize agent states
        self.remaining_planning_length = self.T
        self.num_agents = self.init_agents
        self.agent_states = []

        # Initialize agents at time step 0
        self._initialize_agents()

        self.next_agent_id = self.num_agents

        return self.agent_states


if __name__ == "__main__":
    import sys
    env = VitalSignEnv("models/uganda.npz")

    # reset
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    print(f"Initial info: {info}")

    # action
    obs, reward, terminated, truncated, info = env.step(0)
    print(f"State: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    sys.exit(0)
