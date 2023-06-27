# %%
import numpy as np
from aenet import AdaptiveElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from CML_tool.oracle_model_eval import compute_metrics
import time

# %%
# Import dataset
file  = np.load(
    '/home/barbem4/projects/Initial SLE dataset/Linear models SLE/Synthetic data/Data/oracle_dataset_1/n751_dt50_dn1950_corrpercent0_nstd0.05_corr0_TMcorrN/data.npz')
for key in file:
    exec(key + " = file['" + key + "']")

# %%
# define the models
params_AdaNet = {'C': 1.8532061684132424, 'l1_ratio': 0.873397646239548, 'nu': 0.002759762332391751, 'gamma': 1}
params_ENet = params_AdaNet.copy()
del params_ENet['gamma']
del params_ENet['nu']

# %%
ENet = LogisticRegression(
    penalty = 'elasticnet',
    warm_start=True, 
    max_iter=4000,
    solver = 'saga',
    tol = 1e-4
    )

# %%
AdaNet = AdaptiveElasticNet(
    AdaNet_solver_verbose=False,
    AdaNet_solver = 'default',
    SIS_method = 'one-less',
    refinement=5,
    warm_start=True, 
    max_iter=4000,
    force_solver=False,
    printing_solver = False,
    rescale_EN=False,
    eps_constant=1e-15,
    tol = 1e-4)

# %%
AdaNet.set_params(**params_AdaNet)
ENet.set_params(**params_ENet)

# %%
X_train, X_HOS, y_train, y_HOS = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# X_train = X_train[:,0:60]
# y_train = y_train[:]

# X_HOS = X_HOS[:,0:60]
# y_HOS = y_HOS[:]

# %%
start_time_ENet = time.time()
ENet.fit(X_train, y_train)
end_time_ENet = time.time()
print("ENet execution time:", end_time_ENet-start_time_ENet, "seconds")
# %%
start_time_AdaNet = time.time()
AdaNet.fit(X_train, y_train)
end_time_AdaNet = time.time()
print("AdaNet execution time:", end_time_AdaNet-start_time_AdaNet, "seconds")

# %%
# print the number of nonzero features
print('ENet nonzero coefficients: ', sum(ENet.coef_.ravel()!=0))
print('AdaNet-ENet nonzero coefficients: ', sum(AdaNet.enet_coef_.ravel()!=0))
print('AdaNet nonzero coefficients: ', sum(AdaNet.coef_.ravel()!=0))


# %%
# Test of coefficients
print('ENet coefficient cummulative difference: ',sum(abs(ENet.coef_.ravel()-AdaNet.enet_coef_.ravel())))
print('ENet-AdaNet coefficient cummulative difference: ',sum(abs(ENet.coef_.ravel()-AdaNet.coef_.ravel())))

# %%
# Test accuracy

print('AUROC: ENet-sklearn({}), Ada-ENet({}) and AdaNet({})'.format(
roc_auc_score(y_true=y_HOS, y_score=ENet.predict_proba(X_HOS)[:,1]),
roc_auc_score(y_true=y_HOS, y_score=AdaNet.predict_proba_ENet(X_HOS)[:,1]),
roc_auc_score(y_true=y_HOS, y_score=AdaNet.predict_proba(X_HOS)[:,1])
))
print('AUCPR: ENet-sklearn({}), Ada-ENet({}) and AdaNet({})'.format(
    average_precision_score(y_true=y_HOS, y_score=ENet.predict_proba(X_HOS)[:,1]),
    average_precision_score(y_true=y_HOS, y_score=AdaNet.predict_proba_ENet(X_HOS)[:,1]),
    average_precision_score(y_true=y_HOS, y_score=AdaNet.predict_proba(X_HOS)[:,1])
))

# %%
# Test oracle properties
AdaNet_metrics_dict = compute_metrics(
    true_model_coefs = file['beta_test'],
    model_coefs = AdaNet.coef_.ravel()
    )
ENet_metrics_dict = compute_metrics(
    true_model_coefs = file['beta_test'],
    model_coefs = ENet.coef_.ravel()
    )
print("ENet: ", ENet_metrics_dict)
print("AdaNet: ", AdaNet_metrics_dict)
# %%

print(AdaNet.AdaNet_solver)
# %%
