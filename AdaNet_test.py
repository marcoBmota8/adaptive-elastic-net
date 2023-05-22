# %%
import numpy as np
from aenet_2 import AdaptiveElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.model_selection import train_test_split

# %%
# Import dataset
file  = np.load(
    '/home/barbem4/projects/Initial SLE dataset/Linear models SLE/Synthetic data/Data/oracle_dataset_1/n100_dt50_dn1975_corrpercent0.15_nstd0_corr0.05_TMcorrN/data.npz')
for key in file:
    exec(key + " = file['" + key + "']")
# %%
# define the models
params_AdaNet = {'C':1,'l1_ratio':0.5, 'gamma':1}
params_ENet = params_AdaNet.copy()
del params_ENet['gamma']

ENet = LogisticRegression(
    penalty = 'elasticnet',
    warm_start=True, 
    max_iter=4000,
    solver = 'saga',
    tol = 1e-4
    )

AdaNet = AdaptiveElasticNet(
    AdaNet_solver_verbose=True,
    warm_start=True, 
    max_iter=4000,
    printing_solver = True
)

AdaNet.set_params(**params_AdaNet)
ENet.set_params(**params_ENet)
# %%

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
print('Mean ENet coefficient difference: ',np.mean(abs(ENet.coef_-AdaNet.enet_coef_)))
print('Mean ENet-AdaNet coefficient difference: ',np.mean(abs(ENet.coef_-AdaNet.coef_)))

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
