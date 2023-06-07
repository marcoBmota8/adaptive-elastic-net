# %%
import numpy as np
from aenet import AdaptiveElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from CML_tool.oracle_model_eval import compute_metrics

# %%
# Import dataset
file  = np.load(
    # '/home/barbem4/projects/Initial SLE dataset/Linear models SLE/Synthetic data/Data/oracle_dataset_1/n1000_dt15_dn0_corrpercent0_nstd0_corr0_TMcorrN/data.npz')
    '/home/barbem4/projects/Initial SLE dataset/Linear models SLE/Synthetic data/Data/oracle_dataset_2/n1000_dt15_dn10_corrpercent0_nstd0_corr0_TMcorrN/data.npz')
for key in file:
    exec(key + " = file['" + key + "']")
# %%
# define the models
params_AdaNet =  {'C': 1000.2942264723246768, 'l1_ratio': 0.29408329644256426, 'nu': 0.4605830639847419, 'gamma': 0.5}
params_ENet = params_AdaNet.copy()
del params_ENet['gamma']
del params_ENet['nu']

ENet = LogisticRegression(
    penalty = 'elasticnet',
    warm_start=True, 
    max_iter=1000,
    solver = 'saga',
    tol = 1e-6
    )

# %%
AdaNet = AdaptiveElasticNet(
    AdaNet_solver_verbose=True,
    AdaNet_solver = 'default',
    warm_start=True, 
    max_iter=1000,
    printing_solver = True,
    tol = 1e-8)

AdaNet.set_params(**params_AdaNet)
ENet.set_params(**params_ENet)

# %%
X_train, X_HOS, y_train, y_HOS = train_test_split(X_test, y_test, test_size=0.3, random_state=42)

# X_train = X_train[0:10,0:10]
# y_train = y_train[0:10]

# %%
ENet.fit(X_train, y_train)

# %%
AdaNet.fit(X_train, y_train)
# %%
# Test of coefficients
print('ENet coefficient cummulative difference: ',sum(abs(ENet.coef_.ravel()-AdaNet.ENet.coef_.ravel())))
print('ENet-AdaNet coefficient cummulative difference: ',sum(abs(ENet.coef_.ravel()-AdaNet.coef_.ravel())))

# %%
# Test accuracy

print('AUROC: ENet-sklearn({}), Ada-ENet({}) and AdaNet({})'.format(
roc_auc_score(y_true=y_HOS, y_score=ENet.predict_proba(X_HOS)[:,1]),
roc_auc_score(y_true=y_HOS, y_score=AdaNet.ENet.predict_proba(X_HOS)[:,1]),
roc_auc_score(y_true=y_HOS, y_score=AdaNet.predict_proba(X_HOS)[:,1])
))
print('AUCPR: ENet-sklearn({}), Ada-ENet({}) and AdaNet({})'.format(
    average_precision_score(y_true=y_HOS, y_score=ENet.predict_proba(X_HOS)[:,1]),
    average_precision_score(y_true=y_HOS, y_score=AdaNet.ENet.predict_proba(X_HOS)[:,1]),
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
