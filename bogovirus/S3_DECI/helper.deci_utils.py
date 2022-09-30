import numpy as np
import pandas as pd
import os, sys
import torch
import json
import networkx as nx
from pyvis.network import Network



import causica
from causica.causica.experiment.steps.step_func import load_data
from causica.causica.models.deci.deci import DECI
# from causica.causica.experiment.run_context import RunContext
from causica.causica.datasets.dataset import Dataset

# from data_preprocessing_util import create_variables_json
from causica.causica.datasets.variables import Variables

# DECI configurations
model_config = {
    "tau_gumbel": 0.25,
    "lambda_dag": 10.0, #100
    "lambda_sparse": 1.0, #5.0
    "lambda_prior": 0.0,
    # Choosing a Gaussian DECI model, alternative is "spline"
    "base_distribution_type": "gaussian",
    "imputation": False,
    "spline_bins": 8,
    "var_dist_A_mode": "enco",
    "mode_adjacency": "learn",
    # Maryam added
    "norm_layers": True,
    "res_connection": True,
}

training_params = {
    # Setting higher learning rates can speed up training, but risks some instability
    "learning_rate": 1e-3,
    "var_dist_A_learning_rate": 1e-2,
    "batch_size": 512,
    # This standarizes the data before training. The DECI model operates in standardized space.
    "standardize_data_mean": False,
    "standardize_data_std": False,
    "rho": 1.0,
    "safety_rho": 1e18,
    "alpha": 0.0,
    "safety_alpha": 1e18,
    "tol_dag": 1e-9,
    "progress_rate": 0.65,
    # We are setting this large to wait until we find a DAG
    "max_steps_auglag": 200,
    # We are setting this large to learn a more accurate model.
    "max_auglag_inner_epochs": 2000,
    "max_p_train_dropout": 0.0,
    "reconstruction_loss_factor": 1.0,
    "anneal_entropy": "noanneal",
}

def split_train_test(df, target_cols, data_dir=None, random_state=42):
    from sklearn.model_selection import train_test_split
    target = target_cols #
    X_cols = [col for col in df.columns if col not in target]
    y_col = target

    # train-test-split
    train_X, test_X, train_y, test_y = train_test_split(
        df[X_cols], df[y_col], test_size=0.2, random_state=random_state)

    # train
    df_train = pd.DataFrame(train_X)
    df_train[target] = train_y

    # test - not passed to deci
    df_test = pd.DataFrame(test_X)
    df_test[target] = test_y

    # concatenate 
    df = pd.concat([df_train, df_test])
    df = df.reset_index(drop=True)

    # save data for deci
    if data_dir:
        df_train.to_csv(data_dir + '/bogo/all.csv', index=False, header=False)
        df_test.to_csv(data_dir + '/bogo/test.csv', index=False, header=False)
        df.to_csv(data_dir + 'bogo/all_data.csv',index=False,header=True)

    return df, df_train, df_test

# Method for creating variables.json
def create_variables_json(df, categorical_vars, variables_json_path, text_vars=None, target_column_names=None):
    if text_vars is None:
        text_vars = []
    variables_info = []

    for ident, column_name in enumerate(df.columns):
        query = column_name not in target_column_names if target_column_names is not None else ident != len(df.columns) - 1
        var = {
            "id": ident,
            "query": query,
            "name": column_name,
        }
        if column_name not in text_vars:
            var["type"] = "categorical" if column_name in categorical_vars else "continuous"
            var["lower"] = 0 if column_name in categorical_vars else np.nanmin(df[column_name])
            var["upper"] = (
                categorical_vars[column_name] - 1 if column_name in categorical_vars else np.nanmax(df[column_name])
            )
        elif column_name in text_vars:
            var["type"] = "text"
            var["overwrite_processed_dim"] = 768  # Sentence Transformer model has that dimension

        if column_name in target_column_names:
            var["group_name"]="targets"
        variables_info.append(var)
    variables = Variables.create_from_dict({"variables": variables_info, "metadata_variables": []})
    variables.save(variables_json_path)

def load_deci_model(df, model_name, experiment_dir, variables_json_path, device):
    # Load metadata telling us the data type of each column
    with open(variables_json_path) as f:
        variables = json.load(f)

    # Set up data in a suitable form for DECI to consume, using the loaded data types
    numpy_data = df.to_numpy()
    data_mask = np.ones(numpy_data.shape)
    variables = Variables.create_from_data_and_dict(numpy_data, data_mask, variables)
    dataset = Dataset(train_data=numpy_data, train_mask=np.ones(numpy_data.shape), variables=variables)
    
    # load model
    model = DECI(model_name, dataset.variables, experiment_dir, device, **model_config)
    model.load_state_dict(torch.load(experiment_dir + "/model.pt"))

    # 
    networkx_graph = model.networkx_graph()
    variable_name_dict = {i: var.name for (i, var) in enumerate(variables)}

    return model, networkx_graph, variable_name_dict
    
def run_deci_train(df, model_name, experiment_dir, variables_json_path, device):
    # Load metadata telling us the data type of each column
    with open(variables_json_path) as f:
        variables = json.load(f)

    # Set up data in a suitable form for DECI to consume, using the loaded data types
    mask_data = ~df.isna()
    data_mask = mask_data.to_numpy().astype(int) # np.ones(numpy_data.shape)
    
    # na to zero
    df=df.fillna(0)
    numpy_data = df.to_numpy()
    
    variables = Variables.create_from_data_and_dict(numpy_data, data_mask, variables)
    dataset = Dataset(train_data=numpy_data, train_mask=np.ones(numpy_data.shape), variables=variables)

    # model
    model = DECI.create(model_name, experiment_dir, dataset.variables, model_config, device=device)

    # train
    model.run_train(dataset, training_params)

    # # 
    networkx_graph = model.networkx_graph()
    variable_name_dict = {i: var.name for (i, var) in enumerate(variables)}
    
    return model, networkx_graph, variable_name_dict

def run_deci_train_with_constraints(df, constraint_matrix, model_name, experiment_dir, variables_json_path, device):
    #  Load metadata telling us the data type of each column
    with open(variables_json_path) as f:
        variables = json.load(f)

    # Set up data in a suitable form for DECI to consume, using the loaded data types
    mask_data = ~df.isna()
    data_mask = mask_data.to_numpy().astype(int) # np.ones(numpy_data.shape)

    # na to zero
    df=df.fillna(0)
    numpy_data = df.to_numpy()

    variables = Variables.create_from_data_and_dict(numpy_data, data_mask, variables)
    dataset = Dataset(train_data=numpy_data, train_mask=np.ones(numpy_data.shape), variables=variables)

    # model
    model = DECI.create(model_name, experiment_dir, dataset.variables, model_config, device=device)
    model.set_graph_constraint(constraint_matrix)

    # train
    model.run_train(dataset, training_params)

    # # 
    networkx_graph = model.networkx_graph()
    variable_name_dict = {i: var.name for (i, var) in enumerate(variables)}

    return model, networkx_graph, variable_name_dict

# -----------------------------Visualization----------------------------------------- #
def compute_deci_average_treatment_effect(model, dataset):
	train_data = pd.DataFrame(dataset.train_data_and_mask[0])
	treatment_values = train_data.mean(0) + train_data.std(0)
	reference_values = train_data.mean(0) - train_data.std(0)


	print(train_data.shape)
	ates = []
	for variable in range(treatment_values.shape[0]):
		intervention_idxs = torch.tensor([variable])
		intervention_value = torch.tensor([treatment_values[variable]])
		reference_value = torch.tensor([reference_values[variable]])
		print(f"Computing the ATE between X{variable}={treatment_values[variable]} and X{variable}={reference_values[variable]}")
		# This estimate uses 200 samples for accuracy. You can get away with fewer if necessary
		ate, _ = model.cate(intervention_idxs, intervention_value, reference_value, Ngraphs=1, Nsamples_per_graph=300, most_likely_graph=True)
		ates.append(ate)
	ate_matrix = np.stack(ates)
	return ate_matrix

def calculate_vote_conf(adj_matrix):
	return adj_matrix.sum(0) / adj_matrix.shape[0]

# from statsmodels.stats.proportion import proportion_confint
# #calculate 95% confidence interval with 56 successes in 100 trials
# # proportion_confint(count=56, nobs=100, alpha=0.05, method='normal')

# def calculate_avg_conf(adj_matrix, alpha=0.05, method='normal'):
#     adj_matrix_conf_low = proportion_confint(count=adj_matrix.sum(0), nobs=adj_matrix.shape[0], alpha=alpha, method=method)[0]
#     adj_matrix_conf_high = proportion_confint(count=adj_matrix.sum(0), nobs=adj_matrix.shape[0], alpha=alpha, method=method)[1]
#     adj_matrix_conf_avg = ((adj_matrix_conf_low + adj_matrix_conf_high) /2.0).round(2)
#     return adj_matrix_conf_avg

# ---------------------------------------------------------------------- #
def get_dataset_variables(df, variables_json_path):
    # dataset
    with open(variables_json_path) as f:
        variables = json.load(f)
    numpy_data = df.to_numpy()
    data_mask = np.ones(numpy_data.shape)
    variables = Variables.create_from_data_and_dict(numpy_data, data_mask, variables)
    dataset = Dataset(train_data=numpy_data, train_mask=np.ones(numpy_data.shape), variables=variables)
    return dataset, variables

def save_deci_unweighted_graph(model, variable_name_dict, experiment_dir):
    # pip install pyvis
    from pyvis.network import Network
    
    graph = model.networkx_graph()
    net = Network(notebook=True, directed=True)
    net.from_nx(graph)


    for var_idx, var_name in variable_name_dict.items():
        net.nodes[var_idx]['label'] = var_name
    # labeled_graph = nx.relabel_nodes(graph, variable_name_dict)
    net.show_buttons(filter_=['physics'])
    net.show(os.path.join(experiment_dir, 'unweighted_model_graph.html'))
    
def calc_weighted_graph(model, df, variables_json_path, experiment_dir, samples=200):
    # df + json to DECI dataset & variables
    dataset, variables = get_dataset_variables(df, variables_json_path)

    # multi - sample
    adj_mat = model.get_adj_matrix(samples=200, most_likely_graph=False, do_round=False)
    adj_conf = calculate_vote_conf(adj_mat)
    ate_mat = compute_deci_average_treatment_effect(model, dataset)

    deci_graph = nx.convert_matrix.from_numpy_matrix(adj_conf, create_using=nx.DiGraph)
    deci_ate_graph = nx.convert_matrix.from_numpy_matrix(ate_mat, create_using=nx.DiGraph)

    for n1, n2, d in deci_graph.edges(data=True):
        ate = 0.0 if not (n1,n2) in deci_ate_graph.edges() else deci_ate_graph.get_edge_data(n1, n2)['weight']
        d['confidence'] = d.pop('weight', None)
        d['weight'] = ate
    deci_graph.edges(data=True)
    return deci_graph

def save_edges_to_excel(deci_weighted_graph, variable_name_dict, experiment_dir):
    # set the weights 
    weighted_edges = [(variable_name_dict[_from],variable_name_dict[_to], _prop) for _from, _to, _prop in deci_weighted_graph.edges(data=True)]
    weighted_edges = sorted(weighted_edges, key=lambda x: (abs(x[2]['confidence']), abs(x[2]['weight'])), reverse=True)

    froms = [x[0] for x in weighted_edges]
    tos = [x[1] for x in weighted_edges]
    weights = [x[2]['weight'] for x in weighted_edges]
    confs = [x[2]['confidence'] for x in weighted_edges]

    # save spreadsheet
    weight_df = pd.DataFrame({'from':froms, 'to':tos, 'weight':weights, 'confidence':confs})
    weight_df.to_excel(experiment_dir + "/deci_relations.xlsx", sheet_name="causal_relations", index=False)
    print("Saved in ", experiment_dir + "/deci_relations.xlsx")

def get_nx_graph(deci_graph, variable_name_dict):
    labeled_graph = nx.relabel_nodes(deci_graph, variable_name_dict)
    return nx.draw(labeled_graph, with_labels = True)

def save_deci_weighted_graph(deci_weighted_graph, variable_name_dict, experiment_dir, conf_threshold=0.7):
    # save pyvis
    net = Network(notebook=True, directed=True)
    net.add_nodes(variable_name_dict.values())

    # weighted edges
    for _from, _to, _prop in deci_weighted_graph.edges(data=True):
        conf = _prop['confidence']
        weight = _prop['weight']
        if conf > conf_threshold:
            net.add_edge(variable_name_dict[_from], variable_name_dict[_to], value=_prop['weight'])
        # else:
        #     deci_weighted_graph.remove_edge(variable_name_dict[_from], variable_name_dict[_to])
        
    net.show_buttons(filter_=['physics'])
    net.show(experiment_dir + '/weighted_trimmed_graph.html')
    # net.add_nodes(list(df.columns)) # label=list(df.columns))