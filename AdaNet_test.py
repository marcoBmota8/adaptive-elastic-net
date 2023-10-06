# %%
import numpy as np
from aenet.aenV2 import AdaptiveElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from CML_tool.oracle_model_eval import compute_metrics
from CML_tool.Utils import read_pickle
import pandas as pd
import time

# %%
# Import dataset
file  = np.load(
    '/home/barbem4/projects/Initial SLE dataset/Linear models SLE/Synthetic data/Data/oracle_dataset_1/n751_dt50_dn1950_corrpercent0_nstd0.05_corr0_TMcorrN/data.npz')
for key in file:
    exec(key + " = file['" + key + "']")

# %%
# define the models
params_AdaNet = {'C': 1.5e-1,
 'l1_ratio': 0.3160815048309521,
 'nu': 1.232511206135,
 'gamma': 1}
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
    eps_constant=1e-8,
    nonzero_tol='default',
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
data_SLE = read_pickle(path = '/home/barbem4/projects/Data/Initial Data', filename='marco_ids_final_dates_df_no_unknowns.pkl')
labels = read_pickle(path = '/home/barbem4/projects/Data/Initial Data', filename='SLE_binary_labels_no_unknowns.pkl')
features_names = np.load('/home/barbem4/projects/Data/Initial Data/raw_names.npy', allow_pickle=True)
meta_df = read_pickle(path = '/home/barbem4/projects/Data/Initial Data', filename = 'meta.pkl')

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
    eps_constant=1e-8,
    nonzero_tol='default',
    tol = 1e-4)

params_AdaNet = {'C': 598.530099253463, 'l1_ratio': 0.4718544435940898, 'nu': 19.760031869591373, 'gamma': 1}

AdaNet.set_params(**params_AdaNet)
# %%
AdaNet.fit(data_SLE, labels)

# %%
# print('ENet: ', np.array(features_names)[np.where(AdaNet.enet_coef_.ravel() != 0)[-1]])
aucs = cross_val_score(
    estimator = AdaNet,
    X = data_SLE.values,
    y = labels.values,
    scoring = 'roc_auc',
    cv = 10
)
print('10-fold CV: ', np.mean(aucs))
print('AdaNet: ', np.array(features_names)[np.where(AdaNet.coef_.ravel() != 0)[-1]])

# %%
