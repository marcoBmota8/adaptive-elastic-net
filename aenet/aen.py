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
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class AdaptiveElasticNet(LogisticRegression, MultiOutputMixin, ClassifierMixin):
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
    -AdaNet_solver: What cvxpy solver to use to minimize the AdaNet loss function. By default is the option 'default' that leaves cvxpy teh option to use the most
    appropriate. User can input any solver. If either 'default' or user-defined solver fails the model tries to use either 'ECOS' or 'SCS' to find a 
    suboptimal solution. If none of those works, it reports zero coefficients (trivial solution). (default = 'default')
    -AdaNet_solver_verbose: print out the verbose of the AdaNet solver: cvxpy solver ECOS. (default = False)
    -positive: Positive constraint on coefficients (default = False)
    -positive_tol: When positive = True, cvxpy optimization may still return slightly negative values. 
        If coef > -positive_tol, ignore this and forcively set negative coef to zero.
        Otherwise, raise ValueError.(default = 1e-4)
    -eps_coef: tiny constant to prevent zero division in b_j ** (-gamma) (default = 1e-21)
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
        AdaNet_solver = 'default',
        tol=1e-4,
        positive=False,
        positive_tol=1e-4,
        random_state=None,
        eps_coef=1e+21,
        warm_start = False,
        printing_solver = False,
        n_jobs = 1
    ):
        #inherit parameters from LogisticRegression class 
        super().__init__(
            C = C,
            l1_ratio= l1_ratio,
            solver=solver_ENet,
            penalty='elasticnet',
            fit_intercept=fit_intercept,
            random_state=random_state,
            max_iter=max_iter,
            warm_start=warm_start,
            n_jobs = n_jobs,
            tol = tol,
            )
        self.solver_ENet = solver_ENet
        #AdaNet specific parameters
        self.gamma = gamma
        self.copy_X = copy_X
        self.positive = positive
        self.positive_tol = positive_tol
        self.eps_coef = eps_coef
        self.rescale_EN = rescale_EN # Whether or not to apply Zou and Hastie 2005 rescaling of the ENet coefficients (Naive EN vs. EN)
        #(this distinction was dropped in Friedman et al. 2010 however.)
        self.AdaNet_solver = AdaNet_solver
        self.printing_solver = printing_solver # whether or not to print the solver details on the convergence
        self.AdaNet_solver_verbose = AdaNet_solver_verbose

        #TODO modify this class loss function to support not having an intercept
        if not self.fit_intercept:
            raise NotImplementedError

    def fit(self, X, y):

        if not isinstance(self.l1_ratio, numbers.Number) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1; " f"got l1_ratio={self.l1_ratio}"
            )

        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.enet_coef_, self.weights_ = self._ae(X, y)

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
        n,m = X.shape

        #Fit naive ElasticNet
        self.ENet = self.elastic_net().fit(X, y)
        enet_coef = self.ENet.coef_

        if self.fit_intercept:
            m=m+1
            X_exp = np.c_[X,np.ones(n)]
            enet_coef = np.c_[enet_coef,self.ENet.intercept_]
        
        #Zou and Hastie 2005 rescaling of ElasticNet coefficients to naive ElasticNet coefficients
        if self.rescale_EN:
            EN_rescale_ctant = 1+((1-self.l1_ratio)/(2*self.C))
            enet_coef = EN_rescale_ctant*enet_coef
            
        beta_variables = cvxpy.Variable(m)
        model_prediction = X_exp @ beta_variables

        #Loss
        cross_entropy_loss = self.C * cvxpy.sum(-cvxpy.multiply(model_prediction, y) + cvxpy.logistic(model_prediction)) #Negative log likelihood

        weights = 1.0 / (np.maximum((np.abs(enet_coef))**self.gamma, self.eps_coef))

        print(weights)

        l1_coefs = self.l1_ratio * weights
        l2_coefs = (1 - self.l1_ratio) * 0.5
        l1_penalty = cvxpy.Constant(l1_coefs) @ cvxpy.abs(beta_variables)
        l2_penalty = cvxpy.Constant(l2_coefs) * cvxpy.sum_squares(beta_variables)

        # Positive coefficients constraint
        constraints = [b >= 0 for b in beta_variables] if self.positive else []

        # --- optimization ---
        #/2 * n_samples to make it consistent with sklearn
        problem = cvxpy.Problem(
            objective = cvxpy.Minimize((1/(2*self.n_samples))*(cross_entropy_loss + l1_penalty + l2_penalty)),
            constraints=constraints
        ) 

        try:
            if self.AdaNet_solver == 'default':
                problem.solve(warm_start=True)
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.AdaNet_solver)
                problem.solve(**solver_dict)
        except (ValueError, cvxpy.error.SolverError):
            if self.printing_solver:
                logging.warning(
                    'Selected solver failed. Using alternative options. Check solver and solver_stats for more details')
            solvers = ['ECOS', 'SCS']
            for solver in solvers:
                solver_dict = self._cvxpy_solver_options(solver=solver)
                try:
                    problem.solve(**solver_dict)
                    if 'optimal' in problem.status:
                        self.AdaNet_solver = solver
                        break
                except (ValueError, cvxpy.error.SolverError):
                    continue
            
            self.solver_stats = problem.solver_stats
            
        if self.printing_solver:
            if problem.status in ["infeasible", "unbounded"]:
                logging.warning('Optimization problem status failure')
            elif problem.status not in ["infeasible", "unbounded"]:                 
                logging.info("Problem status was {} used {}% of max iterations = {}".format(
                    problem.status,
                    np.round(problem.solver_stats.num_iters/self.max_iter, decimals = 1),
                    self.max_iter
                    ))
        try:
            beta_sol = beta_variables.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0
        except:
            beta_sol = np.zeros(np.shape(beta_variables))
            if self.printing_solver:
                logging.warning('Optimization failed: Coefficients were set to zero')


        intercept, coef = np.array([beta_sol[-1]]), beta_sol[:-1]
        coef = coef.reshape(1,self.n_features)

        # Check if constraint violation is less than positive_tol.
        if self.positive and self.positive_tol is not None:
            if not all(c.value(self.positive_tol) for c in constraints):
                raise ValueError(f"positive_tol is violated. coef is:\n{coef}")
            
        return (coef, intercept, enet_coef, weights)

    def _cvxpy_solver_options(self, solver):
        if solver == 'ECOS':
            solver_dict = dict(solver=solver,
                                abstol = self.tol,
                                verbose = self.AdaNet_solver_verbose,
                                warm_start = self.warm_start
                               )
        elif solver == 'SCS':
            solver_dict = dict(solver=solver,
                               max_iters=self.max_iter,
                               eps = self.tol,
                               verbose = self.AdaNet_solver_verbose,
                               warm_start = self.warm_start
                               )
        else:
            solver_dict = dict(solver=solver)

        return solver_dict