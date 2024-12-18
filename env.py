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
import random

'''
from gymnasium import Env
'''

## Reconstruct the gmm and minmax object
data = np.load('models/uganda.npz')
means = np.array(data['means'])
covariances = np.array(data['covariances'])
weights = np.array(data['weights'])


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
    ''''Class to simulate the environment 
    for the online RL agent'''

    def _init_(self, path, num_agents, budget, max_num_agents, T, t_min, t_max, joining_rate, leaving_rate):

        '''
        Parameters:
            path: path to the gmm and minmax data
            num_agents: number of agents in the beginning
            budget: the # of medical device available
            max_num_agents: the max # of agents that could possibly appear at the same time
            T: time horizon
            t_min: minimum number of time to wear the device
            t_max: maximum number of time to wear the device
            joining_rate: the rate of people who join  (parameter for Poisson distribution)
            leaving_rate: teh rate of people who leaves (parameter for Poisson distribution)
        '''
        ## If the budget is larger than the max_num_agents, then there is no 
        ## scarcity and no need for decision making
        assert max_num_agents > budget, "Enough device for all patients, no need to train"

        ## According to the rule, all incoming agents should receive the device
        assert budget > num_agents, "Not enough device to allocate at the beginning"

        # load GMM
        self._load_gmm(path)

        ## We are in a finite time horizon
        self.remaining_planning_length = T
        ## Number of agents at the initial time step
        self.num_agents = num_agents
        self.max_num_agents = max_num_agents
        self.budget = budget
        self.t_min = t_min
        self.t_max = t_max
        self.joining_rate = joining_rate
        self.leaving_rate = leaving_rate
        self.next_agent_id = num_agents  ## The id to the first patient arrive in the next round

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
        self.agent_states = []
   
        ## Initialize agents at time step 0
        self._initialize_agents()

    def _initialize_agents(self):
        '''Initialize the agents' states at time step 0'''
        for agent_id in range(self.num_agents):
            ## Sample the initial state for each agent from the mixture
            ## of Gaussian models in IAAI paper
            state,component,mean,cov = sample_agent(self.gmm,self.min_max,self.given_indices)

            ## All incoming agents must receive the device 
            device_flag = 1
            time_joined = 1
            state_with_flag = np.append(state, device_flag)
            overall_state = np.append(state_with_flag, time_joined)
            new_agent_info =  {"id": agent_id, "state": overall_state, "mean": mean, "cov": cov}
            self.agent_states.append(new_agent_info)
            
    def _load_gmm(self, path: str) -> GaussianMixture:
        data = np.load(path)
        means = np.array(data['means'])
        covariances = np.array(data['covariances'])
        weights = np.array(data['weights'])
        scaler_min = np.array(data['scaler_min'])
        scaler_max = np.array(data['scaler_max'])
        names = list((data['names']))

        min_max = {name: [min_val, max_val] for name, min_val, max_val in zip(names, scaler_min, scaler_max)}
        
        # Reconstruct the GMM
        gmm = GaussianMixture(n_components=len(weights), covariance_type='full')

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

    def step(self, action):
        '''
        Parameters: 
            action: a list of the index of arms that's going to be pulled
        '''

        num_joining_agents= np.random.poisson(lam=self.joining_rate, size=1)

        ## All new patients should receive a device
        assert self.budget >= num_joining_agents, "Not enough device for every incoming patient"
        ## Budget constraint cannot be violated
        assert self.budget >= len(action), "Not enough device to allocate according to the policy"

        ## Reduce planning horizon by 1
        self.remaining_planning_length -= 1
        overall_reward_this_step = 0

        '''
            Question left for Mauricio, This part is a bit confusing to me. The action is directly passed in to 
            the step function, then we determine how many agents join. Then it's unclear to me how can we guarantee 
            that all incoming patients will receive a device. 
        '''

        for i in range(len(self.agent_states)):
            agent_info = self.agent_states[i]
            agent_id = agent_info['id']
            overall_state = agent_info['state']
            mean = agent_info['mean']
            cov = agent_info['cov']

            if (i in action):
                if (overall_state[-2] == 1) & (overall_state[-1] >= self.t_max):
                    print(f"Warning, patient {agent_id} has received device for more than {self.t_max} rounds")
                elif (overall_state[-2] == 0):
                    print(f"Warning: You are assigning patient {agent_id} for more than one time")
                overall_state[-2] = 1  # assign device

                ## Update the state under active action
                past_state = overall_state[:6]
                new_state, reward = simulate_one_step(past_state,self.min_max,intervention=True, mean = mean, cov = cov)
                overall_state[:6] = new_state
                overall_reward_this_step += reward
            else:
                if (overall_state[-2] == 1) & (overall_state[-1] < self.t_min):
                    print(f"Warning, patient {agent_id} has not received device by at least {self.t_min} rounds")
                overall_state[-2] = 0
                ## Update the state under active action
                past_state = overall_state[:6]
                ## Update the state under passive action
                new_state, reward = simulate_one_step(past_state,self.min_max,intervention=False, mean = mean, cov = cov)
                overall_state[:6] = new_state
                overall_reward_this_step += reward

            ## add 1 to the joining time
            overall_state[-1] += 1 
            ## update the dictionary
            self.agent_states[i]['state'] = overall_state
    
        # Handle patients who leave at the end
        num_leaving_agents= np.random.poisson(lam=self.leaving_rate, size=1)
        assert num_leaving_agents <= self.num_agents, "More leaving patients than remaining patients"

        # Randomly select patients to delete
        agents_to_delete = random.sample(self.agent_states, num_leaving_agents)

        # Remove the selected elements
        for agent in agents_to_delete:
            self.agent_states.remove(agent)

        ## Update the number of agents
        self.num_agents = self.num_agents + num_joining_agents
        assert self.num_agents <= self.max_num_agents, "Exceeds the max number of agents"

        for i in range(num_joining_agents):
            state,component,mean,cov = sample_agent(self.gmm,self.min_max,self.given_indices)
            device_flag = 1
            time_joined = 1
            state_with_flag = np.append(state, device_flag)
            overall_state = np.append(state_with_flag, time_joined)

            agent_id = self.next_agent_id
            self.next_agent_id += 1
            new_agent_info =  {"id": agent_id, "state": overall_state, "mean": mean, "cov": cov}
            self.agent_states.append(new_agent_info)

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
        
        # Reinitialize agent states
        self.agent_states = []
   
        # Initialize agents at time step 0
        self._initialize_agents()

        self.next_agent_id = num_agents

        return self.agent_states

        

#  def step(self, action, leaving_agents_id_lst, num_joining_agents):
#         '''
#         Parameters: 
#             action: a list of the index of arms that's going to be pulled
#             leaving_agent_id_lst: List of agent IDs to remove
#             num_joining_agents: Number of new agents joining
        
#         '''

#         # leaving_agents_id_lst = np.random.choice(self.agent_states.keys(), )


#         ## All new patients should receive a device
#         assert self.budget >= num_joining_agents, "Not enough device for every incoming patient"
#         ## Budget constraint cannot be violated
#         assert self.budget >= len(action), "Not enough device to allocate according to the policy"

#         ## Reduce planning horizon by 1
#         self.remaining_planning_length -= 1
#         overall_reward_this_step = 0

#         ## Update the state of agents based on who leaves and who joins
#         self.num_agents = self.num_agents - len(leaving_agents_id_lst) + num_joining_agents
#         assert self.num_agents <= self.max_num_agents, "Exceeds the max number of agents"
#         # if self.num_agents > self.max_num_agents:
#         #     print("Number of agents exceeds the budget. Stopping the process.")
#         #     done = True
#         #     reward = -1
#         #     info = {"error": "Exceeded max number of agents"}
#         #     return self.agent_states, reward, done, info
    
#         action = sorted(action)
#         expected_last_elements = list(range(self.num_agents - num_joining_agents + 1, self.num_agents+1))
#         assert action[-num_joining_agents:] == expected_last_elements, \
#             f"Invalid Assignment: Not all incoming patients receive device"
            
#         ## Now we handle the incoming and leaving agents
#         # Step 1: Remove leaving agents
#         for agent_id in leaving_agents_id_lst:
#             if agent_id in self.agent_states:
#                 del self.agent_states[agent_id]

#         # Step 2: Reindex the remaining agents, update the step and reward
#         updated_agent_states = {}
#         new_index = 1
#         for agent_id in sorted(self.agent_states.keys()):
#             agent_id = self.agent_states[agent_id]['id']
#             overall_state = self.agent_states[agent_id]['state']
#             mean = self.agent_states[agent_id]['mean']
#             cov = self.agent_states[agent_id]['cov']

#             if (new_index in action):
#                 ## If get assigned device
#                 if (overall_state[-2] == 1) & (overall_state[-1] >= self.t_max):
#                     print(f"Warning, patient {new_index} has received device for more than {self.t_max} rounds")
#                 elif (overall_state[-2] == 0):
#                     print(f"Warning: You are assigning patient {new_index} for more than one time")
#                 overall_state[-2] = 1

#                 ## Update the state under active action
#                 current_state = overall_state[:6]
#                 ''''
#                     Here I didn't upload the mean and cov here
#                     Maybe I should add it tomorrow
#                 '''
#                 next_state, reward = simulate_one_step(current_state,min_max,intervention=True, mean = mean, cov = cov)
#                 overall_state[:6] = next_state
#                 overall_reward_this_step += reward
#             else:
#                 if (overall_state[-2] == 1) & (overall_state[-1] < self.t_min):
#                     print(f"Warning, patient {new_index} has not received device by at least {self.t_min} rounds")
#                 overall_state[-2] = 0

#                 ## Update the state under passive action
#                 '''
#                     Here I didn't upload the mean and cov here
#                     Maybe I should add it tomorrow
#                 '''
#                 next_state, reward = simulate_one_step(current_state,self.min_max,intervention=False, mean = mean, cov = cov)
#                 overall_state[:6] = next_state
#                 overall_reward_this_step += reward

#             ## add 1 to the joining time
#             overall_state[-1] += 1 

#             ## Change the index

#             ''' TO DO tuple to dict'''
#             updated_agent_states[new_index] = (new_index, overall_state, mean, cov)
#             new_index += 1
        
#         # Step 3: Add new agents with sequential indices
#         for i in range(num_joining_agents):
#             state,component,mean,cov = sample_agent(self.gmm,self.min_max,self.given_indices)
#             device_flag = 1
#             time_joined = 1
#             state_with_flag = np.append(state, device_flag)
#             overall_state = np.append(state_with_flag, time_joined)
#             ''' TO DO tuple to dict'''
#             updated_agent_states[new_index] = (new_index, overall_state, mean, cov)
#             new_index += 1
            
   
#         ## Update self.agent_states with the new dictionary
#         self.agent_states = updated_agent_states

#         # Check if the planning is done
#         if self.planning_length <= 0:
#             done = True
#         else:
#             done = False
        
#         # Set place holder for info
#         info = {}

#         return self.agent_states, overall_reward_this_step, done, info


    