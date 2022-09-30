# gym environment for the bogovirus offline RL learner
# JMA 26 sept 2022

import numpy as np
import pandas as pd
from numpy.random import default_rng
import gym
from gym.spaces import Box, Dict
import faiss

N_NEIGHBORS = 3   # k for nearest neighbors search
SEED = None       # Use a random seed each run
HISTORY_FILE = "patient_data_random_doseE6.csv"
# NROWS = 1000     # subsample the data

class BogoEnv(gym.Env):
    'An environment class derived from the gym environment'

    def __init__(self) -> None:
        'Call this once, and reuse it for all episodes'
        self.n_neighbors = N_NEIGHBORS
        # TODO fix the type warnings for these
        self.action_space = Dict({"Dose": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)})
        self.observation_space = Dict({"Infection": Box(low=0, high=150, shape=(1,), dtype=np.float32), 
                    "Severity":Box(low=0, high=170, shape=(1,), dtype=np.float32),
                    "CumulativeDrug":Box(low=0, high=1, shape=(1,), dtype=np.float32)})
        # State variables.
        self.Infection = None
        self.Severity = None
        self.CumulativeDrug = None

    def reset(self, *, sd=SEED, options=None) -> dict:
        'Initialize an episode'
        super().reset(seed=sd)
        self.my_rng = default_rng(seed=SEED)
        # State variables.
        # THe state is observable, so we use observation as the state. 
        # Of course for a constant policy observability is moot. 
        self.Infection = self.my_rng.integers(low=20, high=40, size=1)[0]
        self.Severity = self.my_rng.integers(low=10, high=30, size=1)[0]
        self.CumulativeDrug = 0
        # Load the history dataset. 
        # names=['infection_prev','severity','cum_drug_prev','infection','drug', 'cum_drug','severity_next','toxicity','outcome'] 
        self.h = pd.read_csv(HISTORY_FILE, header=0, index_col=0)
        # create the nearest neighbor index. 
        #  Remove variables not used in the model
        self.h.drop(['patient_id',  'cohort',  'day_number', 'toxicity'], axis=1, inplace=True)
        #  Features for the predictor -- representing the current state. Only those features samples will be searched on. 
        self.predictor = self.h[['infection_prev', 'severity', 'cum_drug_prev', 'drug']].values.astype('float32')
        self.knn_index = faiss.IndexFlatL2(self.predictor.shape[1])   # build the index
        if not self.knn_index.is_trained:
            print('ERROR, knn index training failed')
        self.knn_index.add(np.ascontiguousarray(self.predictor).astype('float32'))  # faiss is picky about types
        info = {}
        return self._get_ob(), info

    def step(self, action):
        'Increment the state at each stage in an episode, and terminate on death or recovery.'
        terminated = False # reached a final state - dead or recovered
        reward = -1.0    # default penalty for advancing an additional step
        # Die if the drug concentration or severity is high
        if self.CumulativeDrug ** 6 + self.Severity/125 > 1   :
            reward = -200.0
            terminated = True
        # If the patient didn't die, but outlasted the infection, recover. 
        elif self.Infection >= 100:
            reward = 100.0
            terminated = True
        # Should we apply the drug to the CumulativeDrug?  No
        # Or just predict the next observaton from the dataset? Yes
        # In faiss data is in the form of numpy arrays. 
        # Find the k neighbors of ['infection_prev', 'severity', 'cum_drug_prev', 'drug']
        target = np.array([self.Infection, self.Severity, self.CumulativeDrug, action]).reshape((1,4)).astype('float32')
        d_sqr, id = self.knn_index.search(target, self.n_neighbors)
        neighbors = self.h.iloc[id.tolist()[0],:]    # Dont ask why, its alright. 
        # average the values.
        self.Infection = neighbors['infection'].mean()
        self.Severity = neighbors['severity_next'].mean()
        self.CumulativeDrug = neighbors['cum_drug'].mean()
        info = {}
        return self._get_ob(), reward, terminated, False , info

    def _get_ob(self):
        'A convenience function to format the state output '
        return {"Infection":self.Infection, "Severity":self.Severity, "CumulativeDrug":self.CumulativeDrug}
