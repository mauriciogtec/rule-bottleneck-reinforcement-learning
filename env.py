## Create gym environment here.
import os 
import numpy as np
import torch
from gymnasium import Env, spaces
from gymnasium.spaces import Discrete, Box
from sklearn.mixture import GaussianMixture
from new_simulator import sample_agent
from new_simulator import simulate_one_step
import math

'''
from gymnasium import Env
'''

## Reconstruct the gmm and minmax object
data = np.load('models/uganda.npz')
means = np.array(data['means'])
covariances = np.array(data['covariances'])
weights = np.array(data['weights'])

'''
This reading should be improved
'''

scaler_min = np.array([30.7737552, 6.41025641, 71.])
scaler_max = np.array([234.1586382, 53.96608722, 100.])
names = np.array(['PULSE_RATE', 'RESPIRATORY_RATE', 'SPO2'])
# Close the data file
data.close()
# Reconstruct the GMM
gmm = GaussianMixture(n_components=len(weights), covariance_type='full')

# Manually set the parameters
gmm.weights_ = weights
gmm.means_ = means
gmm.covariances_ = covariances

# Compute the precisions_cholesky_ required by the GaussianMixture object
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covariances))

# Create the min_max dictionary
min_max = {name: [min_val, max_val] for name, min_val, max_val in zip(names, scaler_min, scaler_max)}

vital_signs = ['PULSE_RATE',  'RESPIRATORY_RATE','SPO2']
given_indices=[j for j in range(len(vital_signs))]


t_min = 3
t_max = 25


class VitalSignEnv(Env):
    ''''Class to simulate the environment 
    for the online RL agent'''

    def _init_(self, num_agents, budget, max_num_agents, T):

        '''
        num_agents: number of agents in the beginning
        budget: the # of medical device available
        max_num_agents: the max # of agents that could possibly appear at the 
                        same time
        T: time horizon
        '''
        ## If the budget is larger than the max_num_agents, then there is no 
        ## scarcity and no need for decision making
        assert max_num_agents > budget, "Enough device for all patients, no need to train"

        ## According to the rule, all incoming agents should receive the device
        assert budget > num_agents, "Not enough device to allocate at the beginning"

        ## We are in a finite time horizon
        self.total_horizon = T
        self.remaining_planning_length = T
        ## Number of agents at the initial time step
        self.num_agents = num_agents
        self.max_num_agents = max_num_agents
        self.budget = budget

        ## Actions is a list of index that denotes agents who are pulled
        total_action = math.comb(max_num_agents, budget)
        self.action_space = spaces.Box(low = 0, high =1, shape=(total_action,), 
                                        dtype = int)

        '''
            The observation space is 
            (patient agent id [bounded by maximum number of agents],
            # of vital signs plus variance + binary flag about device allocation + time since joined)
        '''
        self.observation_space = spaces.Box(low = 0, high = 1, shape=(max_num_agents, 8), 
                                        dtype=float)
        
        ## Track agent states
        self.agent_states = {}
   
        ## Initialize agents at time step 0
        self._initialize_agents()

    def _initialize_agents(self):
        '''Initialize the agents' states at time step 0'''
        for agent_id in range(self.num_agents):
            ## Sample the initial state for each agent from the mixture
            ## of Gaussian models in IAAI paper
            state,component,mean,cov = sample_agent(gmm,min_max,given_indices)

            ## All incoming agents must receive the device 
            device_flag = 1
            time_joined = 1
            state_with_flag = np.append(state, device_flag)
            overall_state = np.append(state_with_flag, time_joined)
            self.agent_states[agent_id] = (agent_id, overall_state)
        
    def step(self, action, leaving_agents_id_lst, num_joining_agents):
        '''
        Parameters: 
            action: a list of the index of arms that's going to be pulled
            leaving_agent_id_lst: List of agent IDs to remove
            num_joining_agents: Number of new agents joining
        
        '''

        ## All new patients should receive a device
        assert self.budget >= num_joining_agents, "Not enough device for every incoming patient"
        ## Budget constraint cannot be violated
        assert self.budget >= len(action), "Not enough device to allocate according to the policy"

        ## Reduce planning horizon by 1
        self.remaining_planning_length -= 1
        overall_reward_this_step = 0

        ## Update the state of agents based on who leaves and who joins
        self.num_agents = self.num_agents - len(leaving_agents_id_lst) + num_joining_agents
        assert self.num_agents <= self.max_num_agents, "Exceeds the max number of agents"
        # if self.num_agents > self.max_num_agents:
        #     print("Number of agents exceeds the budget. Stopping the process.")
        #     done = True
        #     reward = -1
        #     info = {"error": "Exceeded max number of agents"}
        #     return self.agent_states, reward, done, info
    
        action = sorted(action)
        expected_last_elements = list(range(self.num_agents - num_joining_agents + 1, self.num_agents+1))
        assert action[-num_joining_agents:] == expected_last_elements, \
            f"Invalid Assignment: Not all incoming patients receive device"
            
        ## Now we handle the incoming and leaving agents
        # Step 1: Remove living agents
        for agent_id in leaving_agents_id_lst:
            if agent_id in self.agent_states:
                del self.agent_states[agent_id]

        # Step 2: Reindex the remaining agents, update the step and reward
        updated_agent_states = {}
        new_index = 1
        for agent_id in sorted(self.agent_states.keys()):
            agent_id, overall_state = self.agent_states[agent_id]
            if (new_index in action):
                ## If get assigned device
                if (overall_state[-2] == 1) & (overall_state[-1] >= t_max):
                    print(f"Warning, patient {new_index} has received device for more than {t_max} rounds")
                elif (overall_state[-2] == 0):
                    print(f"Warning: You are assigning patient {new_index} for more than one time")
                overall_state[-2] = 1

                ## Update the state under active action
                current_state = overall_state[:6]
                ''''
                    Here I didn't upload the mean and cov here
                    Maybe I should add it tomorrow
                '''
                next_state, reward = simulate_one_step(current_state,min_max,intervention=True)
                overall_state[:6] = next_state
                overall_reward_this_step += reward
            else:
                if (overall_state[-2] == 1) & (overall_state[-1] < t_min):
                    print(f"Warning, patient {new_index} has not received device by at least {t_min} rounds")
                overall_state[-2] = 0

                ## Update the state under passive action
                '''
                    Here I didn't upload the mean and cov here
                    Maybe I should add it tomorrow
                '''
                next_state, reward = simulate_one_step(current_state,min_max,intervention=False)
                overall_state[:6] = next_state
                overall_reward_this_step += reward

            ## add 1 to the joining time
            overall_state[-1] += 1 

            ## Change the index
            updated_agent_states[new_index] = (new_index, overall_state)
            new_index += 1
        
        # Step 3: Add new agents with sequential indices
        for i in range(num_joining_agents):
            state,component,mean,cov = sample_agent(gmm,min_max,given_indices)
            device_flag = 1
            time_joined = 1
            state_with_flag = np.append(state, device_flag)
            overall_state = np.append(state_with_flag, time_joined)
            updated_agent_states[new_index] = overall_state
            new_index += 1
   
        ## Update self.agent_states with the new dictionary
        self.agent_states = updated_agent_states

        # Check if the planning is done
        if self.planning_length <= 0:
            done = True
        else:
            done = False
        
        # Set place holder for info
        info = {}

        return self.agent_states, overall_reward_this_step, done, info


    def render(self):
        pass

    def reset(self, num_agents, T):
        self.remaining_planning_length = T
        self.num_agents = num_agents
        # Track agent states
        self.agent_states = {}
   
        ## Initialize agents at time step 0
        self._initialize_agents()

        return self.agent_states

        




    