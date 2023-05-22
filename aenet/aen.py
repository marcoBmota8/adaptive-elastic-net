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
from sklearn.base import MultiOutputMixin,ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


class AdaptiveElasticNet(LogisticRegression,ClassifierMixin, MultiOutputMixin):
    """
    Args:
    -C: (default = 1)
    -l1_ratio: (default = 0.5)
    -gamma: (default =  1.0)
    -rescale_EN: Whether or not to apply Zou and Hastie 2005 rescaling of naive ElasticNet coefficients (default = False)
    -fit_intercept: (default = True)
    -max_iter: (default = 4000)
    -tol: Absolute tolerance/accuracy in the solution (coefficients) that is passed to both Elastic Net and Adaptive Elastic Net solvers.(default = 1e-4)
    -solver_ENet: (default = 'saga')
    -AdaNet_solver_verbose: print out the verbose of the AdaNet solver: cvxpy solver ECOS. (default = False)
    -positive: Positive constraint on coefficients (default = False)
    -positive_tol: When positive = True, cvxpy optimization may still return slightly negative values. 
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.(deafult = 1e-4)
    -eps_coef: Small constant to prevent zero division in b_j ** (-gamma) (default = 1e-6)
    -random_state:(default = None)
    -warm_start: (default = False)
    -printing_solver: Whether or not to print the details of the optimization solver solution (default = False)
    -n_jobs: (default = 1)
    -copy_X (default: True)
    
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
    """

    def __init__(
        self,
        C=1.0,
        *,
        l1_ratio=0.5,
        gamma=1.0,
        rescale_EN = False,
        fit_intercept=True,
        max_iter=4000,
        copy_X=True,
        solver_ENet = 'saga',
        AdaNet_solver_verbose = False,
        tol=1e-4,
        positive=False,
        positive_tol=1e-4,
        random_state=None,
        eps_coef=1e-6,
        warm_start = False,
        printing_solver = False,
        n_jobs = 1
    ):

        self.C = C
        self.l1_ratio = l1_ratio
        self.gamma = gamma
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.copy_X = copy_X
        self.positive = positive
        self.positive_tol = positive_tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.eps_coef = eps_coef
        self.solver_ENet = solver_ENet
        self.warm_start = warm_start
        self.rescale_EN = rescale_EN # Whether or not to apply Zou and Hastie 2005 rescaling of the ENet coefficients (Naive EN vs. EN)
        #(this distinction was dropped in Friedman et al. 2010 however.)
        self.printing_solver = printing_solver # whether or not to print the solver details on the convergence
        self.AdaNet_solver_verbose = AdaNet_solver_verbose
                      
        super().__init__(
            C = self.C,
            l1_ratio= self.l1_ratio,
            solver=self.solver_ENet,
            penalty='elasticnet',
            fit_intercept=self.fit_intercept,
            random_state=self.random_state,
            max_iter=self.max_iter,
            warm_start=self.warm_start,
            n_jobs = self.n_jobs,
            tol = self.tol,
            )

        #TODO modify this class loss function to support not having an intercept
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
            self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.enet_coef_, self.weights_ = self._ae(X, y)

        self.dual_gap_ = np.array([np.nan])
        self.n_iter_ = 1

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict(X)
    
    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict_proba(X)
    
    def score(self):
        '''
        Default metric is naive accuracy (highly unwanted but that is Sklearn policy)
        '''
        return super().score()

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
        elastic_net = LogisticRegression(penalty='elasticnet', solver = self.solver_ENet, warm_start=self.warm_start,)


        for k, v in self.get_params().items():
            try:
                elastic_net = elastic_net.set_params(**{k: v})
            except ValueError:
                # ElasticNet does not expect parameter `gamma`
                pass
        if len(params) != 0:
            elastic_net = elastic_net.set_params(**params) # if additional params are given it overides those provided by AdaNet

        return elastic_net

    def _ae(self, X, y):
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

        self.n_samples, self.n_features = X.shape
        beta_variables = [cvxpy.Variable(self.n_features)]
        model_prediction = 0.0

        if self.fit_intercept:
            beta_variables = [cvxpy.Variable(1)]+beta_variables
            ones = cvxpy.Constant(np.ones((self.n_samples,1)))
            model_prediction += ones @ beta_variables[0]

        model_prediction += X @ beta_variables[1]

                # --- define objective function ---
        #   l1 weights w_i are identified with coefs in usual elastic net
        #   l2 weights nu_i are fixed to unity in adaptive elastic net

        #/2 * n_samples to make it consistent with sklearn (asgl uses /n_samples)
        cross_entropy_loss =  -(self.C/(2*self.n_samples))*cvxpy.sum(
            cvxpy.multiply(y, model_prediction) - cvxpy.logistic(model_prediction)
        )
        
        self.ENet = self.elastic_net().fit(X, y)
        enet_coef = self.ENet.coef_

        #Zou and Hastie 2005 rescaling of ElasticNet coefficients to naive ElasticNet coefficients
        if self.rescale_EN == True:
            enet_coef = (1+(1-self.l1_ratio)/(2*self.C))*enet_coef

        weights = 1.0 / (np.maximum((np.abs(enet_coef))** self.gamma, self.eps_coef))

        l1_coefs = self.l1_ratio * weights
        l2_coefs = (1 - self.l1_ratio) * 0.5
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables[1])
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables[1])

        # Positive coefficients constraint
        constraints = [b >= 0 for b in beta_variables] if self.positive else []

        # --- optimization ---
        problem = cvxpy.Problem(
            cvxpy.Minimize(cross_entropy_loss + l1_penalty + l2_penalty), constraints=constraints
        )

        problem.solve(solver='ECOS', 
                      max_iters=self.max_iter,
                      abstol = self.tol,
                      verbose = self.AdaNet_solver_verbose
                      )
        
        if self.printing_solver:
            if problem.status != cvxpy.OPTIMAL:
                print("Problem status was {}; Solver did not converge within {} iterations".format(
                    problem.status,self.max_iter))
            if problem.status == cvxpy.OPTIMAL:
                print("Optimal: Iterations used: {}".format(problem.solver_stats.num_iters))

        try:
            beta_sol = np.concatenate([b.value for b in beta_variables], axis=0)
            beta_sol[np.abs(beta_sol) < self.tol] = 0
        except:
            if self.printing_solver:
                print('Optimization failed: Coefficients were set to zero')

            beta_sol = np.zeros(np.shape(beta_variables[1]))

            if self.fit_intercept:
                beta_sol = np.append(beta_sol,0)


        intercept, coef = np.array([beta_sol[0]]), beta_sol[1:]
        # coef = coef.reshape(1,self.n_features)

        # Check if constraint violation is less than positive_tol.
        if self.positive and self.positive_tol is not None:
            if not all(c.value(self.positive_tol) for c in constraints):
                raise ValueError(f"positive_tol is violated. coef is:\n{coef}")
            
        self.solver_stats = problem.solver_stats

        return (coef, intercept, enet_coef, weights)

    