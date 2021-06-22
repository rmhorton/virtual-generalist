# bn_utils.py
import re, time
from tempfile import NamedTemporaryFile
import pandas as pd
import numpy as np

class BN_utils(object):
    ''

    def __init__(self):
        date_label = re.sub(' ', '_', time.asctime())
        date_label = re.sub(':', '-', date_label)
        # file to archive logs for mlflow
        self.tmp_file = NamedTemporaryFile(suffix = date_label+'.txt', delete=False)

    def filter_data(self, df):
        # Remove rows not including variables of interest      
        rcalevel2_filter = self.rca_root_map.keys()
        # The copy() is necessary to avoid a SettingWithCopyWarning warning 
        data = df[df['RCALevel2'].isin(list(rcalevel2_filter))].copy()
        data.loc[:, 'RCALevel2'] = data['RCALevel2'].map(lambda a: self.rca_root_map[a]) 
        data.reset_index(drop=True, inplace=True)
        return data

    def get_rca_freq(self, data):
        'The fractions of each class label.'
        return data['RCALevel2'].value_counts(normalize=True).to_dict()

    def get_signal_freq(self, data, signals):
        ''
        return data[signals].mean(axis=0)

    def get_D(self, df):
        """
        get_D(df): data driven conditional probabilities given the transformed data

        input:
        - df: transformed data with e17 and signals only (pd.DataFrame)

        output:
        - cond_probs_df: d*d matrix of conditional probabilities. d: length of data columns. cond_probs_df(i,j): Prob(i|j). (pd.DataFrame)
        """
        marginal_probs = df.mean(axis=0, skipna=True)

        joint_probs_mtx = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                joint_probs_mtx[i, j] = joint_probs_mtx[j, i] = (df.iloc[:, [i, j]].sum(axis=1) == 2).mean(skipna=True)

        cond_probs_mtx = np.zeros((df.shape[1], df.shape[1]))
        for i in range(df.shape[1]):
            for j in range(df.shape[1]):
                if marginal_probs[j] != 0:
                    cond_probs_mtx[i, j] = joint_probs_mtx[i, j] / marginal_probs[j]
                else:
                    cond_probs_mtx[i, j] = 0

        cond_probs_df = pd.DataFrame(cond_probs_mtx, columns=marginal_probs.index, index=marginal_probs.index)
        return cond_probs_df

    def precision_recall(self, y_true, y_pred, rc):
        ''
        TP = ((pd.Series(y_true) == rc) & (pd.Series(y_pred) == rc)).sum()
        FP =((pd.Series(y_true) != rc) & (pd.Series(y_pred) == rc)).sum()
        TN = ((pd.Series(y_true) != rc) & (pd.Series(y_pred) != rc)).sum()
        FN = ((pd.Series(y_true) == rc) & (pd.Series(y_pred) != rc)).sum()
        precision = float(0) if FP + TP == 0 else TP / (FP + TP)
        recall = float(0) if FN + TP == 0 else TP / (FN + TP)
        F1 = float(0) if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
        return precision, recall, F1
    
    def prlog(self, *k):
        'A print statement that also copies to a log file.'
        print(*k)
        for j in k:
            self.tmp_file.write(bytes(str(j), 'utf-8'))
        self.tmp_file.write(b'\n')
