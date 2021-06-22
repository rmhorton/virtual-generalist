# optimization_model.py
# 12 Jan 2020   lijinyu
# Construct of wrapper of optimization updating function

import numpy as np
import pandas as pd
from scipy.stats import chisquare
from pgmpy.inference import VariableElimination
from bn_utils import get_D, get_signal_freq , aks_rc_signal_map, prlog
# import json
from pathlib import Path
import pandas as pd

# Set to positive values {1..3} for verbose output.
DBG = 3
DATA_PATH = '.'


###################################################################################

class OptimizationModel(object):
    """
    Construct of wrapper of optimization updating function
    """
    def __init__(self, reader, df, signals, iter_limit=10):
        self.iter_limit = iter_limit
        self.reader = reader

        self.model = reader.get_model()
        self.adjacency = list(self.model.adjacency())
        self.parent_dict = self.reader.get_parents()

        self.root_nodes = self.get_root_nodes()
        self.interm_nodes = self.get_interm_nodes()
        self.obs_nodes = self.get_obs_nodes()

        self.df = df
        self.signals = signals

        self.D = get_D(df[signals])
        self.freq = get_signal_freq(df, signals)
        print('D\n', self.D)
        print('Signal freq:\n', self.freq)

    def get_root_nodes(self):
        """
        get root nodes (components) list given the BayesNet.
        """
        return self.model.get_roots()

    def get_interm_nodes(self):
        """
        get intermediate nodes list given the BayesNet.
        """
        interm_nodes = []

        for k, v in self.parent_dict.items():
            if len(v) > 0:
                for item in v:
                    if item not in self.root_nodes:
                        interm_nodes.append(item)
        return interm_nodes

    def get_obs_nodes(self):
        """
        get observation (signal) nodes list given the BayesNet.
        """
        return self.model.get_leaves()

    def get_cond_prob_B(self):
        """
        get_cond_prob_B(self): conditional probabilities given the pre-defined network

        output:
        - B: d*d matrix of conditional probabilities B from BayesNet.
             d: length of obs_nodes. cond_probs_df(i,j): Prob(i|j). (pd.DataFrame)
        """
        obs_nodes = self.obs_nodes
        infer = VariableElimination(self.model)

        cond_prob_mtx = np.zeros((len(obs_nodes), len(obs_nodes)))
        for i in range(len(obs_nodes)):
            for j in range(len(obs_nodes)):
                if i == j:
                    cond_prob_mtx[i, j] = 1
                else:
                    cond_prob_mtx[i, j] = infer.query(variables=[obs_nodes[i]],
                                                      evidence={obs_nodes[j]: 1},
                                                      show_progress=False).values[1]
        B = pd.DataFrame(cond_prob_mtx, columns=obs_nodes, index=obs_nodes)

        return B

    def init_prior(self, root_node):
        """
        initialize the prior probability of root nodes
        initial_prob = LB + (UB-LB) * RC_most_related_signal_prevalence (* pr_factor)
        """
        cpd = self.model.get_cpds(root_node)
        cpd_new_values = cpd.get_values()
        for idx, state in enumerate(cpd.state_names[root_node][1:]):
            if root_node in aks_rc_signal_map.keys():
                try:
                    related_signal = aks_rc_signal_map[root_node]
                    cpd_new_values[idx + 1] = self.LB + (self.UB - self.LB) * self.freq[related_signal]
                    print(root_node, cpd_new_values[idx + 1])
                except:
                    cpd_new_values[idx + 1] = 0.5
                    print(root_node + ' does not have match.')
            else:
                cpd_new_values[idx + 1] = 0.5
                print(root_node + ' does not have match.')

        cpd_new_values[0] = 1 - cpd_new_values[1:].sum()

    def update_prior(self, root_node, pr_factor):
        # TODO: rewrite update_prior function
        """
        Multiple the root_node prior == abnormal by pr_factor in the node CPT.
        """

        cpd_new_values = self.model.get_cpds(root_node).get_values()

        cpd_new_values[1:] = cpd_new_values[1:] * pr_factor
        cpd_new_values[0] = 1 - cpd_new_values[1:].sum()

    def get_node_cpd(self, node):
        """
        get the cpd table (TableauCPD) of the given node.
        """
        return self.model.get_cpds(node)

    def get_values_from_cpd(self, node):
        """
        Get the values (abnormal probabilties from cpd table. Assume a binary observation with abnormal state in the second row.
        """
        cpt = self.get_node_cpd(node)
        return cpt.get_values()[1, :]

    def update_cpd(self, node, state_index, val):
        """
        update cpd values given the state index and new value.
        """
        # if node in self.root_nodes:
        #     self.update_prior(node, val)
        # else:
        cpd = self.get_node_cpd(node)
        cpd_new_values = cpd.get_values()
        cpd_new_values[:, state_index] = 1-val, val

    def objective(self, loss_function='KL'):
        """
        output: error result
        """
        if loss_function == 'BD_diff':
            loss = self.BD_diff()
        elif loss_function == 'KL':
            loss = self.KL_divergence()
        elif loss_function == 'chi_sq':
            loss = self.chi_sq()
        else:
           prlog(f"Error: {loss_function} not found") 
        return loss

    def BD_diff(self):
        """
        objective function of B and D difference 1/n*(n-1)\sum(B-D)^2
        """
        B_new = self.get_cond_prob_B()
        return ((B_new - self.D) ** 2).mean().mean()

    def KL_divergence(self):
        """
        objective function of KL divergence
        """
        def KL(P, Q):
            """ Epsilon is used here to avoid conditional code for
            checking that neither P nor Q is equal to 0. """

            epsilon = 0.00001

            # You may want to instead make copies to avoid changing the np arrays.
            P = P + epsilon
            Q = Q + epsilon

            divergence = np.sum(P * np.log(P / Q))
            return divergence

        B_new = self.get_cond_prob_B()
        return KL(B_new.values.ravel(), self.D.values.ravel())

    def chi_sq(self):
        """
        objective function of chi square statistics
        """
        B_new = self.get_cond_prob_B()
        f_obs = B_new.values.ravel()
        f_exp = self.D.values.ravel()
        print((f_obs - f_exp) ** 2 / f_exp)
        try:
            chisq_stat = ((f_obs - f_exp) ** 2 / f_exp).sum()
        except:
            chisq_stat = 0

        # return chisquare(f_obs, f_exp)[0]
        return chisq_stat

    def evaluate_fun(self, node, state_index, val, loss_function):
        """
        error / objective function given the node and its params
        """
        self.update_cpd(node, state_index, val)
        return self.objective(loss_function)

# EOF
