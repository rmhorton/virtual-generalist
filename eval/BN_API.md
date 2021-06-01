# BN API

## PGMpy

a_model = BayesianModel([(parent, child),..])  -> BayesianModel
a_model.check_model() -> boolean


? = PGM(shape:List)
### Access functions 

a_model.nodes()  -> NodeView

a_model.get_roots() -> List

a_model.get_edges() -> OutEdgeView([(parent, child),...])

a_model/get_cpds() -> List

a_model.local_independencies(variable:str)

All dependent nodes:
model.active_trail_nodes('D', observed='G')  -> {'D': {set of nodes}}

### Update functions

a_cpd = TabularCPD(variable:str, variable_card:int, values:List)
        - or -
a_cpd = TabularCPD(variable='S', variable_card=2,
                      values=[[0.95, 0.2],
                              [0.05, 0.8]],
                      evidence=['I'],
                      evidence_card=[2],
                      state_names={'S': ['Bad', 'Good'],
                                   'I': ['Dumb', 'Intelligent']})

a_model.add_cpds(a_cpd, ...)

###  Inference

infer = VariableElimination(a_model)

x = infer.query(var= [variable, ...], evidence=[(variable, value:int), ...])
    - or -
x = infer.query(var= [variable, ...], evidence=[{variable: value:int), ...})

x.values() -> [factor, ...]  # ? clique potentials? 

jt = a_model.to_junction_tree()

Max A Posteriori inference

q = a_model.map_query(variables=[...], evidence={...})

Forward Sampling

samples_df = BayesianModelSampling(a_model).forward_sample(size=int(1e5))

# Learning

Parameters - learning "in place"

a_model.fit(data=samples_df, estimator=MaximumLikelihoodEstimator)