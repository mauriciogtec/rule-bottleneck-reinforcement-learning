import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
import seaborn as sns
import math
import pandas as pd
import random

degree_of_arm_noise=0.15
intervention_success_rate=0.7
variability_window=5

#Hyperparameters for 3 settings
setting="uganda"
min_entries=1
time_size=20
vital_signs = ['PULSE_RATE',  'RESPIRATORY_RATE','SPO2']
num_comp=5
sample_size=-1
num_timesteps=1

#HELPER
def create_training_dataset3(pivot_df, num_timesteps, time_size, min_entries=1):
    '''
        This function inputs a dataframe, given all pateints in dataframe,
        check how many time series are valid for that patient If valid, add
        that patient to valid_patient_ids list, and return the (input, output data),
        where input is a time series of designated length, and output is the value 
        at the next timestep

    '''

    input_data = []
    output_data = []
    patient_entry_counts = {}  # Dictionary to store the count of valid sequences for each patient_id
    valid_patient_ids = set()  # Set to store patient_ids that meet the min_entries threshold
    patient_rewards = {}

    # Group by 'patient_id' and process each group
    for patient_id, group in pivot_df.groupby('patient_id'):
        group = group.sort_values('generatedat')
        valid_sequences = []  # Temporary list to hold valid sequences for the current patient_id
        rewards = []

        # Iterate through the group to find valid sequences
        for i in range(len(group) - num_timesteps):
            valid_sequence = True
            base_time = group.iloc[i]['generatedat']
            for j in range(1, num_timesteps + 1):
                expected_time = base_time + pd.Timedelta(minutes=time_size * j)
                if group.iloc[i + j]['generatedat'] != expected_time:
                    valid_sequence = False
                    break
            if valid_sequence:
                input_values = group.iloc[i:i + num_timesteps][vital_signs].values.flatten()
                output_values = group.iloc[i + num_timesteps][vital_signs].values
                valid_sequences.append((input_values.astype(float), output_values.astype(float)))
                rewards.append(group.iloc[i + num_timesteps]['reward'])


        patient_rewards[patient_id] = np.mean(rewards)
        # Only add to final dataset if valid_sequences count meets the threshold
        if len(valid_sequences) > min_entries:
            valid_patient_ids.add(patient_id)
            for input_values, output_values in valid_sequences:
                input_data.append(input_values)
                output_data.append(output_values)

    return np.hstack((np.array(input_data), np.array(output_data))), list(valid_patient_ids)

def conditional_sample_gmm(gmm, given_values, given_indices,component_index=None):
    """
    Sample from the conditional distribution of a Gaussian Mixture Model. The state from the
    next time step is sampled from a Guassian mixture model conditional on the state of the
    current step.

    Parameters:
    - gmm: Fitted GaussianMixture object
    - given_values: The values of the given variables
    - given_indices: The indices of the given variables

    Returns:
    - Sample from the conditional distribution
    """
    all_indices = np.arange(gmm.means_.shape[1])
    remaining_indices = np.setdiff1d(all_indices, given_indices)

    # Extract the means and covariances of the components
    means = gmm.means_
    covariances = gmm.covariances_
    weights = gmm.weights_


    # Convert to DataFrame
    df = pd.DataFrame(covariances[0])

    # Print nicely formatted
    #print(df.to_string(index=False, float_format="%.8f"))
    #print(means,weights)
    # Calculate conditional means and covariances for each component
    if not component_index is None:
        mean_given = means[component_index, given_indices]
        mean_remaining = means[component_index, remaining_indices]
        cov_given_given = covariances[component_index][np.ix_(given_indices, given_indices)]
        cov_remaining_given = covariances[component_index][np.ix_(remaining_indices, given_indices)]
        cov_given_remaining = covariances[component_index][np.ix_(given_indices, remaining_indices)]
        cov_remaining_remaining = covariances[component_index][np.ix_(remaining_indices, remaining_indices)]
        #print("means",mean_given,mean_remaining)
        #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

        cov_inv_given_given = np.linalg.inv(cov_given_given)
        conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
        conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

        return np.clip(np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov),0,1)
    else:
      conditional_means = []
      conditional_covs = []
      for k in range(gmm.n_components):
          mean_given = means[k, given_indices]
          mean_remaining = means[k, remaining_indices]
          cov_given_given = covariances[k][np.ix_(given_indices, given_indices)]
          cov_remaining_given = covariances[k][np.ix_(remaining_indices, given_indices)]
          cov_given_remaining = covariances[k][np.ix_(given_indices, remaining_indices)]
          cov_remaining_remaining = covariances[k][np.ix_(remaining_indices, remaining_indices)]
          #print("means",mean_given,mean_remaining)
          #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

          cov_inv_given_given = np.linalg.inv(cov_given_given)
          conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
          conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

          conditional_means.append(conditional_mean)
          conditional_covs.append(conditional_cov)

      # Sample from the conditional distribution of each component
      component_samples = [
          np.random.multivariate_normal(mean=conditional_means[k], cov=conditional_covs[k])
          for k in range(gmm.n_components)
      ]

      # Sample a component based on the weights
      component = np.random.choice(gmm.n_components, p=weights)

    return np.clip(component_samples[component],0,1)

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
    #print(df.to_string(index=False, float_format="%.8f"))
    #print(means,weights)
    # Calculate conditional means and covariances for each component
    mean_given = mean[given_indices]
    mean_remaining = mean[remaining_indices]
    cov_given_given = cov[np.ix_(given_indices, given_indices)]
    cov_remaining_given = cov[np.ix_(remaining_indices, given_indices)]
    cov_given_remaining = cov[np.ix_(given_indices, remaining_indices)]
    cov_remaining_remaining = cov[np.ix_(remaining_indices, remaining_indices)]
    #print("means",mean_given,mean_remaining)
    #print("covariates",cov_given_given,cov_remaining_given,cov_given_remaining,cov_remaining_remaining)

    cov_inv_given_given = np.linalg.inv(cov_given_given)
    conditional_mean = mean_remaining + cov_remaining_given @ cov_inv_given_given @ (given_values - mean_given)
    conditional_cov = cov_remaining_remaining - cov_remaining_given @ cov_inv_given_given @ cov_given_remaining

    return np.clip(np.random.multivariate_normal(mean=conditional_mean, cov=conditional_cov),0,1)

'''
    This function inputs the vital signs data of the patients and cleans the dataset
    It shrinks the vital sign data to the range of [0, 1]
    Use the create training data function, it creates the training dataset with satisfies 
    valid time series criterion and return the list of patient id which is in the 
    valid time series dataset.
    The training data is later used to feed a Guassuan mixture model of 5 components and
    random state 42
'''

def create_model():
    df = pd.read_csv("uganda-60-10--1-SPO2-PULSE_RATE-RESPIRATORY_RATE_full.csv")
    df['generatedat'] = pd.to_datetime(df['generatedat'])
    def to_float(value):
            try:
                return float(value)
            except ValueError:
                return np.nan
    df.loc[:, 'doublevalue'] = df['doublevalue'].apply(to_float)
    df = df.dropna(subset=['doublevalue','patient_id'])
    ## filter out abnormal values (outliers)
    # 
    condition = (df['name'] == 'PULSE_RATE') & (df['doublevalue'] > 300)
    df = df[~condition]

    condition = (df['name'] == 'RESPIRATORY_RATE') & (df['doublevalue'] > 60)
    df = df[~condition]

    condition = (df['name'] == 'SPO2') & ((df['doublevalue'] < 50) | (df['doublevalue'] > 100))
    df = df[~condition]

    condition = (df['name'] == 'COVERED_SKIN_TEMPERATURE') & (df['doublevalue'] > 45)
    df = df[~condition]

    df['generatedat'] = pd.to_datetime(df['generatedat'], errors='coerce')
    df.set_index('generatedat', inplace=True)
    df.sort_index(inplace=True)

    ## first row of each patient id
    first_entries = df.groupby('patient_id').head(1).reset_index()
    ## This line calculates the median of doublevalue for each patient_id and name group, 
    ## resampled by time intervals of time_size minutes.
    median_values = df.groupby(['patient_id','name']).resample(str(time_size)+'T')['doublevalue'].median()
    median_values = median_values.reset_index()

    ## In the pivot table, the patient id and generatedat are rows, and the rest of 
    ## the vital signs are columns
    resampled_df = median_values.dropna(subset=['doublevalue'])
    pivot_df = resampled_df.pivot_table(index=['patient_id', 'generatedat'], columns='name', values='doublevalue')
    pivot_df = pivot_df.dropna().reset_index()

    pivot_df['reward'] = pivot_df.apply(lambda row: reward_function(row[vital_signs].to_dict()), axis=1)

    # for vital_sign in vital_signs:
    #     sns.histplot(pivot_df[vital_sign], kde=True)
    #     plt.title('Distribution of Column Name')
    #     plt.show()

    ## Normalize the vital sign value to between 0 and 1
    min_max={}
    for sign in vital_signs:
        min_max[sign] = [pivot_df[sign].min(),pivot_df[sign].max()]
        pivot_df[sign] = min_max_normalize(pivot_df[sign])

    first_val=['patient_id', 'generatedat', 'reward']
    pivot_df=pivot_df[first_val+vital_signs]
    df_proc = pivot_df[vital_signs]

    idx = pivot_df.groupby('patient_id')['generatedat'].idxmin()
    earliest_entries_df = pivot_df.loc[idx].reset_index(drop=True)
    rev_df = pivot_df.copy()
    clean_data(vital_signs,rev_df,min_max)

    combined_training,valid_patient_ids = create_training_dataset3(pivot_df, num_timesteps, time_size=time_size, min_entries=min_entries)
    # np.save("./preprocessed_files/"+f_name_t, combined_training)
    # np.save("./preprocessed_files/"+f_name_v, valid_patient_ids)
    pivot_df=pivot_df[pivot_df['patient_id'].isin(valid_patient_ids)]
    rev_df=rev_df[rev_df['patient_id'].isin(valid_patient_ids)]

    gmm = GaussianMixture(n_components=num_comp, covariance_type='full', random_state=42)

    gmm.fit(combined_training)

    return gmm,min_max

gmm, min_max = create_model()

#Exp REWARDS
import math

def temperature_penalty(temperature):
    if temperature <= 38:
        return 0
    else:
        return -math.exp(abs(temperature - 38.0)/2)  # Exponential penalty

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

def manual_normalization(x,p_min,p_max):
  return (x-p_min)/(p_max-p_min)

def min_max_normalize(column):
    #print(column.min(),column.max())
    return (column - column.min()) / (column.max() - column.min())

def reverse_min_max_normalize(column, min_val, max_val):
    return column * (max_val - min_val) + min_val

def manual_normalize_data(vital_signs,p_dict,min_max):
  for sign in vital_signs:
    p_dict[sign] = np.clip(manual_normalization(p_dict[sign],min_max[sign][0], min_max[sign][1]),0,1)
  return p_dict

def clean_data(vital_signs,p_df,min_max):
  for sign in vital_signs:
    p_df[sign] = reverse_min_max_normalize(p_df[sign], min_max[sign][0], min_max[sign][1])
  return p_df

def reward_function(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)
  reward=0
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      reward+=temperature_penalty(sign_dict[signs])
    elif signs=="PULSE_RATE":
      reward+=pulse_penalty(sign_dict[signs])
    elif signs=="RESPIRATORY_RATE":
      reward+=respiratory_penalty(sign_dict[signs])
    elif signs=="SPO2":
      reward+=spo2_penalty(sign_dict[signs])
  return reward

''' improve_vital_signs: One model of the positive effect of intervention 
    (assigning a medical device). It reduces the vital signs by a certain amount.

    Input:
        - sign_dict: A dictionary whose key is the vital sign name, and value
            is the corresponding value
        - rev_norm: a binary variable indicating whether you need to clean data 
            in the beginning
        
    Output:
        - sign_dict: A dictionary with vital signs as key and modified value as
            values

'''

def improve_vital_signs(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      if temperature_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-1.5
    elif signs=="PULSE_RATE":
      if pulse_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-15
    elif signs=="RESPIRATORY_RATE":
      if respiratory_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-10
    elif signs=="SPO2":
      if spo2_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]+3
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

''' improve_vital_signs: Another model of the positive effect of intervention 
    (assigning a medical device). The medical staff reacts to the alert 70%
    of time in reality. It reduces the vital signs by a certain amount.

    Input:
        - sign_dict: A dictionary whose key is the vital sign name, and value
            is the corresponding value
        - rev_norm: a binary variable indicating whether you need to clean data 
            in the beginning
        
    Output:
        - sign_dict: A dictionary with vital signs as key and modified value as
            values

'''
def improve_vital_signs1(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  if random.random() < intervention_success_rate:
    for signs in sign_dict:
      if signs=="COVERED_SKIN_TEMPERATURE":
        if temperature_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-1.5
      elif signs=="PULSE_RATE":
        if pulse_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-15
      elif signs=="RESPIRATORY_RATE":
        if respiratory_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-10
      elif signs=="SPO2":
        if spo2_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]+3
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

''' improve_vital_signs: Another model of the positive effect of intervention 
    (assigning a medical device). The abnormal vital sign is probablistically 
    adjusted towards normal

    Input:
        - sign_dict: A dictionary whose key is the vital sign name, and value
            is the corresponding value
        - rev_norm: a binary variable indicating whether you need to clean data 
            in the beginning
        
    Output:
        - sign_dict: A dictionary with vital signs as key and modified value as
            values

'''

def improve_vital_signs2(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  for signs in sign_dict:
    if signs=="COVERED_SKIN_TEMPERATURE":
      if temperature_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(1.5, 0.5)
    elif signs=="PULSE_RATE":
      if pulse_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(15, 5)
    elif signs=="RESPIRATORY_RATE":
      if respiratory_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]-np.random.normal(10, 10/3)
    elif signs=="SPO2":
      if spo2_penalty(sign_dict[signs])<0:
        sign_dict[signs]=sign_dict[signs]+np.random.normal(3, 1)
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

''' improve_vital_signs: Another model of the positive effect of intervention 
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

'''

def improve_vital_signs3(sign_dict,rev_norm=False,o_values=None):
  if rev_norm:
    #print(sign_dict)
    sign_dict=clean_data(vital_signs,sign_dict,o_values)

  #print(sign_dict)
  if random.random() < intervention_success_rate:
    for signs in sign_dict:
      if signs=="COVERED_SKIN_TEMPERATURE":
        if temperature_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(1.5, 0.5)
      elif signs=="PULSE_RATE":
        if pulse_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(15, 5)
      elif signs=="RESPIRATORY_RATE":
        if respiratory_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]-np.random.normal(10, 10/3)
      elif signs=="SPO2":
        if spo2_penalty(sign_dict[signs])<0:
          sign_dict[signs]=sign_dict[signs]+np.random.normal(3, 1)
  sign_dict=manual_normalize_data(vital_signs,sign_dict,o_values)
  return sign_dict

''' interventions: This function models the effect of intervention. if the patient's value 
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
'''


def interventions(current_values,min_max,mean=None,cov=None,given_indices=list(range(len(vital_signs)))):
  if reward_function(dict(zip(vital_signs,current_values)),rev_norm=True,o_values=min_max)>=0:
    return conditional_sample_mnd(current_values, given_indices,mean=mean,cov=cov)
  else:
    #new_signs= conditional_sample_gmm(gmm, current_values, given_indices,component_index=component_index)
    #print("Old", current_values)
    new_signs=improve_vital_signs3(dict(zip(vital_signs,current_values)),rev_norm=True,o_values=min_max)
    #print("NEW",[new_signs[vital] for vital in vital_signs])
    return conditional_sample_mnd([new_signs[vital] for vital in vital_signs], given_indices,mean=mean,cov=cov)
    #return resample_values(gmm,min_max,component_index=component_index)[0]


''' simulate_one_step: based on the current value, calculate what's the next state for vital signs,
    the variance of vital sign for the past five timesteps, and the reward

    Input:
      - current_state: current_state[0] stores the current vital sign, current_state[2] stores the 
        vital signs for the past five timesteps
    
    Output:
      - next_signs: the vital sign for the next timestep
      - variablity: the variance of vital signs from the past five states
      - signs_history: the vital sign history for the past five states
      - reward: the reward for the next signs
'''
def simulate_one_step(current_state,min_max,intervention=False,mean=None,cov=None,given_indices=list(range(len(vital_signs)))):
    current_signs=current_state[0]
    signs_history=current_state[2]
    #print(current_signs)

    if intervention:
      next_signs=interventions(current_values=current_signs,min_max=min_max,mean=mean,cov=cov)
    else:
      next_signs=conditional_sample_mnd(current_signs, given_indices,mean=mean,cov=cov)

    for i in range(len(vital_signs)):
      del signs_history[i][0]
      signs_history[i].append(next_signs[i])

    variability=[np.var(l) for l in signs_history]

    reward=reward_function(dict(zip(vital_signs,next_signs)),rev_norm=True,o_values=min_max)
    return [next_signs,variability,signs_history],reward

''' resample_values: You sample from a multivariate Gaussian for your initial value,
and you sample conditioned on the previous value until you have enough sign history to 
calculate variability

Then you return the current signs, the variability of the past timesteps, the past
vital sign values, and the corresponding reward of the currrent vital sign
'''

def resample_values(min_max,given_indices,mean,cov):
  sample=np.clip(np.random.multivariate_normal(mean=mean,cov=cov),0,1)
  current_signs=[sample[i] for i in given_indices]
  signs_history=[[] for _ in range(len(vital_signs))]
  for i in range(len(vital_signs)):
    signs_history[i].append(sample[i])

  for _ in range(variability_window-1):
    current_signs=conditional_sample_mnd(current_signs, given_indices,mean=mean,cov=cov)
    for i in range(len(vital_signs)):
      signs_history[i].append(current_signs[i])

  #print(signs_history)
  #for l in signs_history:
    #print(l,np.var(l))
  variability=[np.var(l) for l in signs_history]
  #print(variability)
  reward=reward_function(dict(zip(vital_signs,current_signs)),rev_norm=True,o_values=min_max)
  return [current_signs,variability,signs_history],reward

''' sample_agent: you choose a component basesd on weight of each component for the multivariate 
    Gaussian, then you get the sample from it.
    You perturb the vital sign mean and cov by choosing a mean and covariance from another component
    in the mixture model, and randomly sampling a influence factor to determine the magnitude of
    pertubation
'''
def sample_agent(gmm,min_max,given_indices):
  weights = gmm.weights_

  # Normalize the weights to ensure they sum to 1
  weights /= np.sum(weights)

  # Sample an index based on the weights
  component = np.random.choice(len(weights), p=weights)

  means = gmm.means_
  covariances = gmm.covariances_
  mean=means[component]
  cov=covariances[component]
  state, _ = resample_values(min_max,given_indices,mean=mean,cov=cov)

  perturb=random.choice([i for i in range(len(weights)) if i != component])

  x=random.uniform(0, degree_of_arm_noise)
  y=random.uniform(0, degree_of_arm_noise)

  mean=(1-x)*mean+x*means[perturb]
  cov=(1-y)*cov+y*covariances[perturb]

  #print(mean,cov)
  #pertubation
  return state,component,mean,cov

gmm,min_max=create_model()
T=100
given_indices=[j for j in range(len(vital_signs))]

state,component,mean,cov=sample_agent(gmm,min_max,given_indices)
#print(state)


all_indices = np.arange(gmm.means_.shape[1])
remaining_indices = np.setdiff1d(all_indices, given_indices)

# Initialize the array to hold the simulated values
simulated_values = []
simulated_reward = []
simulated_values.append(state)
simulated_reward.append(0)

intervention=False
# Simulate the future values for T timesteps
for t in range(1, T):
    given_values = simulated_values[-1]

    intervention=True
    next_values, reward=simulate_one_step(given_values,min_max,intervention=intervention,mean=mean,cov=cov)
    simulated_values.append(next_values)
    simulated_reward.append(reward)
    #print(next_values)


#print(simulated_values)
#print([s[0] for s in simulated_values])
tmp_df = pd.DataFrame([s[0] for s in simulated_values], columns=vital_signs)
#print(tmp_df)
clean_data(vital_signs,tmp_df,min_max)
#print(tmp_df)
tmp_df['reward'] = simulated_reward
#print(simulated_rewards)

plt.figure(figsize=(10, 6))

#print(group['COVERED_SKIN_TEMPERATURE'].mean(),group['COVERED_SKIN_TEMPERATURE'].std())
# Plot each vital sign
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111)

#print(group['COVERED_SKIN_TEMPERATURE'].mean(),group['COVERED_SKIN_TEMPERATURE'].std())
# Plot each vital sign

for vital in vital_signs:
    ax1.plot(tmp_df.index, tmp_df[vital],label=vital)

ax1.set_xlabel('Time')
ax1.set_ylabel('Vital Sign Values')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create a secondary y-axis for the rewards
ax2 = ax1.twinx()
ax2.plot(tmp_df.index, tmp_df['reward'], label='reward', color='red')
ax2.set_ylabel('Reward')
ax2.legend(loc='upper right')
plt.legend()
plt.grid(True)
plt.show()