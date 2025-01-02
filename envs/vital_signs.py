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
    
class VitalSignEnv(Env):
    """'Class to simulate the environment
    for the online RL agent"""

    def __init__(
        self,
        path: str,
        init_agents = 2,  # B=3 in the paper
        max_num_agents = 10,  # N=20 in the paper
        budget = 2, # They have a budget, which does not necessarily equal to init_agent
        T = 20,  # T = 100 in the paper
        t_min = 1,  # t_min = 3 in the paper
        t_max = 5,  # t_max = 5 in the paper
        joining_number = 2,  # = two patients in the paper, no letter
        system_duration = 10,  # = 50 in the paper, no letter
        joining_interval = 5, ## In the paper, patients join every 5 timesteps
        degree_of_arm_noise=0.15,
        intervention_success_rate=0.7,
        variability_window=5
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
        if (budget > max_num_agents):
            raise ValueError("Enough device for all patients, no need to train")

        # ## According to the rule, all incoming agents should receive the device
        if (init_agents > budget):
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
        observation_space_size = 3 + 3 + 3 * variability_window + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(max_num_agents, observation_space_size), dtype=float
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
                "time_joined": 1
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
        '''read the meam, cov, and min_max of the mixture model'''
        gmm = self.gmm
        min_max = self.min_max
        means = gmm.means_
        covariances = gmm.covariances_
        weights = gmm.weights_

        # Normalize the weights to ensure they sum to 1
        weights /= np.sum(weights)

        # Sample an index based on the weights
        component = np.random.choice(len(weights), p=weights)

        mean = means[component]
        cov = covariances[component]

        return mean,cov, min_max
    
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
        if random.random() < self.intervention_success_rate:
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
        conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (
            vital_values - mean_given
        )
        conditional_cov = (
            cov_remaining_remaining
            - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining
        )

        return np.clip(
            np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov), 0, 1
        )
    
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

        if (
            self._reward_function(
                dict(zip(vital_signs, vital_values))
            )
            >= 0
        ):
            return self._conditional_sample_mnd(vital_values, given_indices)
        else:
            # new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
            # print("Old", current_values)
            new_signs = self._improve_vital_signs3(
                dict(zip(vital_signs, vital_values))
            )
            # print("NEW",[new_signs[vital] for vital in vital_signs])
            return self._conditional_sample_mnd(
                [new_signs[vital] for vital in vital_signs],
                given_indices
            )

    def _simulate_one_step(self, agent_index, intervention = False):
        """simulate_one_step: based on the current value, calculate what's the next state for vital signs,
        the variance of vital sign for the past five timesteps, and the reward
        """

        mean, cov, min_max = self._read_gmm_minmax()
        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        agent_info = self.agent_states[agent_index]
        vital_values = agent_info['vitals']
        signs_history = agent_info['signs_history']

        vital_signs = list(min_max.keys())
        given_indices = np.arange(len(vital_signs))

        if intervention:
            next_signs = self._interventions(vital_values)
        else:
            next_signs = self._conditional_sample_mnd(vital_values, given_indices)

        for i in range(len(vital_signs)):
            del signs_history[i][0]
            signs_history[i].append(next_signs[i])

        variability = [np.var(l) for l in signs_history]

        reward = self._reward_function(
            dict(zip(vital_signs, next_signs))
        )
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

        sample = np.clip(np.random.multivariate_normal(mean=mean, cov=cov), 0, 1)
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
        variability = [np.var(l) for l in signs_history]
        # print(variability)
        reward = self._reward_function(
            dict(zip(vital_signs, current_signs))
        )
        return [current_signs, variability, signs_history], reward

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
        state, _ = self._resample_values()

        perturb = random.choice([i for i in range(len(weights)) if i != component])

        x = random.uniform(0, self.degree_of_arm_noise)
        y = random.uniform(0, self.degree_of_arm_noise)

        mean = (1 - x) * mean + x * means[perturb]
        cov = (1 - y) * cov + y * covariances[perturb]

        # print(mean,cov)
        # pertubation
        return state, component, mean, cov
    
    def _change_state_to_obs(self):
        agent_states = self.agent_states
        # Initialize an empty list to hold rows
        agent_matrix = []
        
        # Iterate over each agent's state
        for agent in agent_states:
            # Extract values from the dictionary
            vitals = agent['vitals']  # 1D array with 3 values
            variability = agent['variability']  # List with 3 elements
            signs_history = np.concatenate(agent['signs_history'])  # Flatten the list of lists
            has_device = [agent['has_device']]  # Scalar value converted to list for concatenation
            time_joined = [agent['time_joined']]  # Scalar value converted to list for concatenation

            # Concatenate all components into a single row
            agent_row = np.concatenate([vitals, variability, signs_history, has_device, time_joined])
            agent_matrix.append(agent_row)
        return np.array(agent_matrix)

    def step(self, action):
        """
        Parameters:
            action: a list of 0 and 1 that determines whether the current agent's device
            will be taken or not. A 1 corresponds to action, namely the agent's device will
            be reallocated to another agent.
        """

        if (len(action) != self.num_agents):
            raise ValueError("Invalid Action: Number of Agents Mismatch")

        self.remaining_planning_length -= 1
        time_passed = self.T - self.remaining_planning_length
        if (time_passed % self.joining_interval == 0):
            num_joining_agents = self.joining_number
        else: 
            num_joining_agents = 0

        ## All new patients should receive a device
        if (num_joining_agents > self.budget):
            raise ValueError("Not enough device for every incoming patient")
        
        ## This variable tracks how many 
        ## devices are kept from last round
        device_kept_from_last_round = 0 

        ## Accumulated reward from this round
        overall_reward_this_step = 0

        for i in range(len(self.agent_states)):
            agent_info = self.agent_states[i]
            agent_id = agent_info["id"]
            time_joined = agent_info["time_joined"]
            has_device = agent_info["has_device"]
            mean = agent_info["mean"]
            cov = agent_info["cov"]

            if (action[i] == 0) and (has_device == 1):
                ## The agent has device in the last round and 
                ## keeps the device
                if (time_joined >= self.t_max):
                    raise ValueError(f"patient {agent_id} has received device for more than {self.t_max} rounds")
                device_kept_from_last_round += 1
                # Update the state under active action
                new_state, reward = self._simulate_one_step(i, intervention = True)
                overall_reward_this_step += reward
            elif (action[i] == 0) and (has_device == 0):
                ## The agent doesn't have device in the last round
                ## and maintain the status
                # Update the state under passive action
                new_state, reward = self._simulate_one_step(i, intervention = False)
                overall_reward_this_step += reward    
            elif (action[i] == 1) and (has_device == 1):
                ## The agent has device in te last round and the 
                ## device got taken away
                if (time_joined < self.t_min):
                    raise ValueError(f"Patient {agent_id} has not received device by at least {self.t_min} rounds")
                # Update the state under active action
                new_state, reward = self._simulate_one_step(i, intevention = False)
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
            self.agent_states[i]['variability'] = new_variability
            self.agent_states[i]['signs_history'] = new_signs_history
            self.agent_states[i]["time_joined"] = time_joined
            self.agent_states[i]["has_device"] = has_device

            if (time_joined >= self.system_duration):
                del self.agent_states[i]
        
        if device_kept_from_last_round + num_joining_agents > self.budget:
            raise ValueError("Not enough device to allocate according to the policy")
    
        ## Update the number of agents
        self.num_agents = self.num_agents + num_joining_agents
        if (self.num_agents > self.max_num_agents):
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

    def reset(self):
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


if __name__ == "__main__":
    import sys
    env = VitalSignEnv("models/uganda.npz")

    # reset
    obs, info = env.reset()
    print(f"Initial state: {obs}")
    print(f"Initial info: {info}")
    
    # action
    obs, reward, terminated, truncated, info = env.step([0, 0])
    print(f"State: {obs}")
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print(f"Info: {info}")

    sys.exit(0)

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