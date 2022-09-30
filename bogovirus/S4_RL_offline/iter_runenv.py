# harness to run BogoEnv gym environment
# JMA 27 Sept 2022
import os, re, sys
import pandas as pd
import numpy as np
import gym
import envs

EPISODE_LEN = 22
HISTORY_FILE = "patient_data_random_doseE6.csv"   # A million simulation records
REPS = 100     # Number of episodes to average 

# def write_episode_history():
#     lbls = ['Reward','Dose', "Infection", "Severity", "CumulativeDrug"] 
#     run = pd.DataFrame(run_trajectory, columns=lbls).iloc[0:last_episode+1,:]
#     print(run)

def one_run(the_dose):
    'Iterate over the range of doses, running "reps" replications for one episode'
    # A column for each variable: reward, Drug-action, Infection, CumulativeDrug, Severity 
    run_trajectory = np.empty((EPISODE_LEN, 5))

    last_episode = EPISODE_LEN
    # Starting stage
    observation, info = bg_env.reset()
    run_trajectory[0] = [0, the_dose] + list(observation.values())

    # Transition until termination
    for i in range(1, EPISODE_LEN):
        observation, reward, terminated, truncated, info = bg_env.step(the_dose)
        run_trajectory[i] = [reward, the_dose] + list(observation.values())
        if terminated or truncated:
            observation, info = bg_env.reset()
            last_episode = i
            break
    #  Return the state and action for each stage in the episode. 
    return run_trajectory[1:(last_episode+1),:]

#### MAIN ########################################################################
bg_env = gym.make('BogoEnv-v0')

run_stats = dict()
for dose in range(0, 15):
    one_run_stats = []   # Accumulate survival values 

    dose = dose/10.0

    dose_log = None     # Accumulate episode state action table. 
    for i in range(REPS):
        obs = one_run(dose)
        # save the episode trajectory
        lbls = ['Reward','Dose', "Infection", "Severity", "CumulativeDrug"] 
        if dose_log is None:
            dose_log = pd.DataFrame(obs, columns=lbls)
        else:
            dose_log = pd.concat([dose_log, pd.DataFrame(obs, columns=lbls)], axis=0)
        fin_r = obs[obs.shape[0]-1, 0]        #the last reward tells recover or die
        live_or_die = int(fin_r/abs(fin_r) if fin_r != 0 else 0)
        one_run_stats.append(live_or_die)
        print('/',end='') # Keep from getting bored. 
    run_stats.update({round(dose,2): one_run_stats})
    dose_log.to_csv(f'trajectory_{dose}.csv')  # Save the trajectories for each dose level. 
    print()

df_run_stats = pd.DataFrame(run_stats)
print('\nstats:\n',df_run_stats.applymap(lambda k: 1 if k==1 else 0).mean(axis=0))   # Survival fraction 
df_run_stats.to_csv('run_stats100.csv')

bg_env.close()

