# %%
import numpy as np
from aenet import AdaptiveElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from CML_tool.oracle_model_eval import compute_metrics
import time

# %%
#Generate data 
X,y = make_classification(
    n_samples = 751,
    n_features = 2000,
    n_informative= 50,
    n_redundant= 40,
    n_repeated = 0,
    n_classes = 2,
    n_clusters_per_class = 1,
    random_state = 42
)

# Generate random weights
weights = np.random.rand(X.shape[1])

# Set some coefficients to zero
fraction_of_zeros = 0.9
# Calculate the number of elements to set to zero
num_zeros = int(fraction_of_zeros * weights.size)

# Generate a boolean mask with True values indicating which elements to set to zero
mask = np.random.choice([True, False], size=weights.size, replace=True, p=[num_zeros/weights.size, 1 - num_zeros/weights.size])

# Set the selected elements to zero
weights[mask] = 0

# Compute the linear response
linear_response = np.dot(X, weights)

# Apply the logistic link function to get y
p = 1 / (1 + np.exp(-linear_response))
# draw binary labels from binomial distribution
y = np.random.binomial(1,p)
# %%
# define the models
params_AdaNet ={'C':2.5,
 'l1_ratio': 0.9,
 'nu':0.0002,
 'gamma': 1}
params_ENet = params_AdaNet.copy()
del params_ENet['gamma']
del params_ENet['nu']

# %%
ENet = LogisticRegression(
    penalty = 'elasticnet',
    warm_start=True, 
    max_iter=2000,
    solver = 'saga',
    tol = 1e-4
    )

# %%
AdaNet = AdaptiveElasticNet(
    AdaNet_solver_verbose=False,
    AdaNet_solver = 'default',
    refinement=5,
    warm_start=True, 
    max_iter=2000,
    force_solver=False,
    printing_solver = False,
    rescale_EN=True,
    eps_constant=1e-6,
    tol = 1e-4)

# %%
AdaNet.set_params(**params_AdaNet)
ENet.set_params(**params_ENet)

# %%
X_train, X_HOS, y_train, y_HOS = train_test_split(X, y, test_size=0.3, random_state=42)

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
print('AdaNet-ENet nonzero coefficients: ', sum(AdaNet.ENet.coef_.ravel()!=0))
print('AdaNet nonzero coefficients: ', sum(AdaNet.coef_.ravel()!=0))


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
    true_model_coefs = weights.ravel(),
    model_coefs = AdaNet.coef_.ravel()
    )
ENet_metrics_dict = compute_metrics(
    true_model_coefs = weights.ravel(),
    model_coefs = ENet.coef_.ravel()
    )
print("ENet: ", ENet_metrics_dict)
print("AdaNet: ", AdaNet_metrics_dict)
# %%

print(AdaNet.AdaNet_solver)
# %%
