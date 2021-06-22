# metrics.py
# Compute the accuracy of a BN model against a labelled dataset

# NOte run this in conda environment azureml
#   > conda activate azureml
import os
import sys
import re
import time
import pandas as pd
import heapq
from sklearn.metrics import confusion_matrix, accuracy_score
from pathlib import Path

# Used to track runs
from mlflow import log_metric, log_param, log_artifact, set_experiment, get_experiment, set_tracking_uri

from optimization_model import VariableElimination
# from bn_utils import *
from pgmpy.readwrite.XMLBIF import XMLBIFReader

import json

# TODO merge BN_utils and metrics classes 

DBG = 3

class Metrics(object):
    def __init__(self, y_true, signals, current_cache, model):
        self.y_true = y_true
        self.signals = signals
        self.model = model
        self.current_cache = current_cache

    def signals_to_dict(self):
        'Those signals that == 1 are set to abnormal. '
        # abnormal_signals = self.signals[self.signals == 1].to_dict()
        # abnormals = {k:'abnormal' for k,v in abnormal_signals.items()}
        abnormal = self.signals.to_dict()
        return abnormal

    def signals_key(self, signals):
        'Convert the signals dict to an immutable to be used as a key.'
        return tuple(signals.values)

    def infer_multiple_root_causes(self):
        ''
        infer = VariableElimination(self.model)
        infer_res = dict()
        roots = self.model.get_roots()
        root_joint = infer.query(variables=roots, evidence=self.signals_to_dict(), show_progress=False)
        root_names = root_joint.scope()
        for j, root in enumerate(root_names):
            summable_axis = list(range(len(root_names)))
            del summable_axis[j]
            infer_res[root] = root_joint.values.sum(axis=tuple(summable_axis))[1]
        return infer_res

    def get_inference(self):
        ''
        if self.signals_key(self.signals) in self.current_cache:
            return self.current_cache[self.signals_key(self.signals)]
        infer = VariableElimination(self.model)
        infer_res = dict()
        for root in self.model.get_roots():
            prob = infer.query(variables=[root], evidence=self.signals_to_dict(), show_progress=False).values
            cpd = self.model.get_cpds(root)
            state_names = cpd.state_names[root]
            for i in range(1, len(state_names)):
                # infer_res[root + '_' + cpd.state_names[root][i]] = prob[i]
                infer_res[root] = prob[i]
        return infer_res

    def top_matched_causes(self,  degree=1):
        # self.root_marginals = self.infer_multiple_root_causes()
        # if all(x < 0.15 for x in list(self.get_inference().values())):
        #     return ['NA']
        # else:
        return heapq.nlargest(degree, self.get_inference().keys(), key=lambda k: self.get_inference()[k])

    def get_key(self, dic, val):
        'Invert a dict - find the key for a value. TODO - use existing utilities.'
        for key, value in dic.items():
            if val in value:
                return key
        return 'no match'

# EOF
