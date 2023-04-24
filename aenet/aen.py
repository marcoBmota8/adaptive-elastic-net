'''
Author: Marco Barbero
Date: April 2023
Description: Adaptive elastic net model classifier
Adapted from: https://github.com/simaki/adaptive-elastic-net -> Regressor (at the time of download (March 2023) had some mistakes)
'''

import numbers
import warnings
import cvxpy
import numpy as np
from asgl import ASGL
from sklearn.base import MultiOutputMixin
from sklearn.base import RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._coordinate_descent import _alpha_grid
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


class AdaptiveElasticNet(ASGL, LogisticRegression, MultiOutputMixin, RegressorMixin):
    """
    Objective function is

        (1 / 2 n_samples) * C * sum_i (-y_i * log(P(y_i_hat)) - (1-y_i)*log(P(1-y_i_hat)))
            +  l1ratio * sum_j |coef_j|
            +  (1 - l1ratio) * sum_j w_j * ||coef_j||^2

            where P(.) = sigmoid(.) = 1/(1+exp(-.))

        w_j = |b_j| ** (-gamma)
        b_j = coefs obtained by fitting ordinary elastic net

        i: sample
        j: feature
        |X|: abs
        ||X||: square norm

    Parameters
    ----------
    - C : float, default=1.0
        Constant that multiplies the penalty terms.
    - l1_ratio : float, default=0.5
        float between 0 and 1 passed to ElasticNet
        (scaling between l1 and l2 penalties).
    - gamma : float > 0, default=1.0
        To guarantee the oracle property, following inequality should be satisfied:
            gamma > 2 * nu / (1 - nu)
            nu = lim(n_samples -> inf) [log(n_features) / log(n_samples)]
        default is 1 because this value is natural in the sense that
        l1_penalty / l2_penalty is not (directly) dependent on scale of features
    - fit_intercept = True
    - max_iter : int, default 10000
        The maximum number of iterations.
    - positive : bool, default=False
        When set to `True`, forces the coefficients to be positive.
    - positive_tol : float, optional
        Numerical optimization (cvxpy) may return slightly negative coefs.
        (See cvxpy issue/#1201)
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.
        If `positive_tol=None` always ignore (default)
    - eps_coef : float, default 1e-6
        Small constant to prevent zero division in b_j ** (-gamma).

    Attributes
    ----------
    - coef_
    - intercept_
    - enet_coef_
    - weights_

    Examples
    --------
    >>> from sklearn.datasets import make_regression

    >>> X, y = make_regression(n_features=2, random_state=0)
    >>> model = AdaptiveElasticNet(random_state=42)
    >>> model.fit(X, y)
    AdaptiveElasticNet(random_state=42, solver='default', tol=1e-05)
    >>> print(model.coef_)
    [14.24414426 48.9550584 ]
    >>> print(model.enet_coef_)
    [18.8... 64.5...]
    >>> print(model.weights_)
    [0.0530... 0.0154...]
    >>> print(model.intercept_)
    2.092...
    >>> print(model.predict([[0, 0]]))
    [2.092...]

    Constraint:
    >>> X, y = make_regression(n_features=10, random_state=0)
    >>> model = AdaptiveElasticNet(positive=True).fit(X, -y)
    >>> model.coef_
    array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , 13.39621927,  0.        ])
    """

    def __init__(
        self,
        C=1.0,
        *,
        l1_ratio=0.5,
        gamma=1.0,
        rescale_EN = False,
        fit_intercept=True,
        precompute=False,
        max_iter=4000,
        copy_X=True,
        solver_Adaptive="OSQP",
        solver_ENet = 'saga',
        tol=None,
        positive=False,
        positive_tol=1e-3,
        random_state=None,
        eps_coef=1e-6,
    ):
        params_asgl = dict(model="lm", penalization="asgl")
        if solver_Adaptive is not None:
            params_asgl["solver"] = solver_Adaptive
        if tol is not None:
            params_asgl["tol"] = tol

        super().__init__(**params_asgl)

        self.C = C
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.precompute = precompute
        self.copy_X = copy_X
        self.positive = positive
        self.positive_tol = positive_tol
        self.random_state = random_state
        self.eps_coef = eps_coef
        self.solver_ENet = solver_ENet
        self.solver_Adaptive = solver_Adaptive
        self.rescale_EN = rescale_EN # Whether or not to apply Zou and Hastie 2005 rescaling of the ENet coefficients (Naive EN vs. EN)
        #(this distinction was dropped in Friedman et al. 2010 however.)

        if not self.fit_intercept:
            raise NotImplementedError

    def fit(self, X, y, check_input=True):
        if abs(self.C) == np.inf:
            warnings.warn(
                "With large C, this algorithm does not converge "
                "well. You are advised to use the LinearRegression "
                "estimator",
                stacklevel=2,
            )

        if not isinstance(self.l1_ratio, numbers.Number) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1; " f"got l1_ratio={self.l1_ratio}"
            )

        if check_input:
            X_copied = self.copy_X and self.fit_intercept
            X, y = self._validate_data(
                X,
                y,
                accept_sparse="csc",
                order="F",
                dtype=[np.float64, np.float32],
                copy=X_copied,
                multi_output=True,
                y_numeric=True,
            )

        self.coef_, self.intercept_, self.enet_coef_, self.weights_ = self._ae(X, y)

        self.dual_gap_ = np.array([np.nan])
        self.n_iter_ = 1

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super(LogisticRegression, self).predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super(LogisticRegression,self).predict_proba(X)
    
    def sigmoid(x):
        return 1/(1+cvxpy.exp(-x))
    
    def score(self):
        '''
        Default metric is naive accuracy (highly unwanted but that is Sklearn policy)
        '''
        return super(LogisticRegression,self).score()

    def elastic_net(self, **params):
        """
        ElasticNet with the same parameters of self.

        Parameters
        ----------
        - **params
            Overwrite parameters.

        Returns
        -------
        elastic_net : ElasticNet
        """
        elastic_net = LogisticRegression(penalty='elasticnet')


        for k, v in self.get_params().items():
            try:
                elastic_net = elastic_net.set_params(**{k: v})
            except ValueError:
                # ElasticNet does not expect parameter `gamma`
                pass
        if len(params) != 0:
            elastic_net = elastic_net.set_params(**params) # if additional params are given it overides those provided by AdaNet

        return elastic_net

    def _ae(self, X, y) -> (np.array, float):
        """
        Adaptive elastic-net counterpart of ASGL.asgl

        Returns
        -------
        (coef, intercept, enet_coef, weights)
            - coef : np.array, shape (n_features,)
            - intercept : float
            - enet_coef : np.array, shape (n_features,)
            - weights : np.array, shape (n_features,)
        """
        check_X_y(X, y)

        n_samples, n_features = X.shape

        if self.fit_intercept:
            beta_variables = [cvxpy.Variable(n_features + 1)] # adds the intercept contribution to the prediction
            X = np.concatenate((X,np.ones((X.shape[0],1))), axis = 1) # adds a columns of ones to the end of the data matrix
        else:
            beta_variables = [cvxpy.Variable(n_features)]
            #_, beta_variables = self._num_beta_var_from_group_index(group_index)
            #beta_variables = cvxpy.Variable()

        # --- define objective function ---
        #   l1 weights w_i are identified with coefs in usual elastic net
        #   l2 weights nu_i are fixed to unity in adaptive elastic net

        # /2 * n_samples to make it consistent with sklearn (asgl uses /n_samples)
        model_prediction = X @ beta_variables
        cross_entropy_loss = -cvxpy.sum(
            y * cvxpy.log(self.signoid(model_prediction)) + (1 - y) * cvxpy.log(1 - self.signoid(model_prediction))) / (2*n_samples)

        enet_coef = self.elastic_net().fit(X, y).coef_

        if self.rescale_EN == True:
            enet_coef = (1+(1-self.l1_ratio)/(2*self.C))*enet_coef

        weights = 1.0 / (np.maximum(np.abs(enet_coef), self.eps_coef) ** self.gamma)

        l1_coefs = self.l1_ratio * weights
        l2_coefs = (1 - self.l1_ratio) * 1.0
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables)
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables)

        constraints = [b >= 0 for b in beta_variables] if self.positive else []

        # --- optimization ---
        problem = cvxpy.Problem(
            cvxpy.Minimize(self.C * cross_entropy_loss+ l1_penalty + l2_penalty), constraints=constraints
        )
        # OSQP seems to be default for our problem.
        problem.solve(solver=self.solver_Adaptive, max_iter=self.max_iter)

        if problem.status != "optimal":
            raise ConvergenceWarning(
                f"Solver did not reach optimum (Status: {problem.status})"
            )

        beta_sol = np.concatenate([b.value for b in beta_variables], axis=0)
        beta_sol[np.abs(beta_sol) < self.tol] = 0

        intercept, coef = beta_sol[0], beta_sol[1:]

        # Check if constraint violation is less than positive_tol.
        if self.positive and self.positive_tol is not None:
            if not all(c.value(self.positive_tol) for c in constraints):
                raise ValueError(f"positive_tol is violated. coef is:\n{coef}")
            
        self.solver_stats = problem.solver_stats

        return (coef, intercept, enet_coef, weights)

    