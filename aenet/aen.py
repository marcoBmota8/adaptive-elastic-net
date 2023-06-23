'''
Author: Marco Barbero
Date: April 2023
Description: Adaptive elastic net model classifier
Adapted from: https://github.com/simaki/adaptive-elastic-net -> Regressor (at the time of download (March 2023) had some mistakes)
'''

import numbers
import warnings
from sklearn.exceptions import ConvergenceWarning
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
    -C: inverse of penalty strength both in ENet and AdaNet loss(default = 1)
    -l1_ratio: L1-L2 relationship in ENet loss(default = 0.5) 
    -nu: AdaNet weighted L1 penalty hypereparameter (default = 1). AdaNet imposses the same L2 penalty in both ENet and AdaNet but allows for different L1 penalty in ENet and AdaNwt loss functions. 
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
    -eps_constant: Small constant to avoid dividing by zero when constructing the adptive weights (default 1e-15).
    -refinement: number of iterative refinement steps for CVXOPT solver. (default = 10)
    -random_state:(default = None)
    -warm_start: (default = False)
    -printing_solver: Whether or not to print the details of the optimization solver solution (default = False)
    -force_solver: Whether to force the usage of the passed solver to optimize AdaNet optimization. 
        If it fails, the oprimization will fail and no attempt will be made with alternative solvers. (default = False)
    -n_jobs: (default = 1)
    -copy_X (default: True)
    
    Objective function is

        (1 / 2 n_samples) * C * sum_i (-y_i * log(P(y_i_hat)) - (1-y_i)*log(P(1-y_i_hat)))
            +  nu * sum_j w_j * |coef_j|
            +  0.5*(1 - l1ratio) * ||coef_j||^2

        w_j = |b_j| ** (-gamma)
        b_j = coefs obtained by fitting ordinary elastic net

        i: sample
        j: nonzero features in ENet model
        |X|: abs (L1 norm)
        ||X||: sqrt(sum of squares) (L2 norm)
    """

    def __init__(
        self,
        C=1.0,
        *,
        l1_ratio=0.5,
        gamma=1.0,
        nu = 1.0,
        rescale_EN = False,
        fit_intercept=True,
        max_iter=4000,
        copy_X=True,
        solver_ENet = 'saga',
        AdaNet_solver_verbose = False,
        AdaNet_solver = 'default',
        eps_constant = 1e-15,
        refinement = 10,
        tol=1e-4,
        positive=False,
        positive_tol=1e-4,
        random_state=None,
        warm_start = False,
        printing_solver = False,
        force_solver = False,
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
        self.nu = nu
        self.copy_X = copy_X
        self.positive = positive
        self.positive_tol = positive_tol
        self.eps_constant = eps_constant # Small constant to avouid dividing by zero when constructing the adaptive weights
        self.rescale_EN = rescale_EN # Whether or not to apply Zou and Hastie 2005 rescaling of the ENet coefficients (Naive EN vs. EN)
        #(this distinction was dropped in Friedman et al. 2010 however.)
        self.AdaNet_solver = AdaNet_solver
        self.printing_solver = printing_solver # whether or not to print the solver details on optimization (convergence warnings, etc)
        self.AdaNet_solver_verbose = AdaNet_solver_verbose
        self.force_solver = force_solver # Whether force to use the selected solver
        self.refinement = refinement # number of iterative refinement steps for CVXOPT solver

        if not self.fit_intercept:
            raise NotImplementedError('AdaNet only suspports models with intercept. Set fit_intercept = True.')


    def fit(self, X, y):

        if not isinstance(self.l1_ratio, numbers.Number) or not 0 <= self.l1_ratio <= 1:
            raise ValueError(
                "l1_ratio must be between 0 and 1; " f"got l1_ratio={self.l1_ratio}"
            )

        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.enet_coef_, self.weights_ = self._ae(np.asarray(X), np.asarray(y))

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict(np.asarray(X))
    
    def predict_proba(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict_proba(np.asarray(X))
    
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
                # ElasticNet does not expect parameter `gamma` o `nu`
                pass
        if len(params) != 0:
            elastic_net = elastic_net.set_params(**params) # if additional params are given it overides those provided by AdaNet

        return elastic_net

    def _ae(self, X, y):
        """
        Returns
        -------
        (coef, intercept, enet_coef, weights)
            - coef : np.array, shape (n_features,)
            - intercept : float (if requested via fit_intercept)
            - enet_coef : np.array, shape (n_features,)
            - weights : np.array, shape (n_features,)
        """
        check_X_y(X, y)

        if not self.printing_solver:
            # Suppress ConvergenceWarning
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            # Suppress CVXPY UserWarning
            warnings.filterwarnings("ignore", category=UserWarning)


        #Fit naive ElasticNet
        self.ENet = self.elastic_net().fit(X, y)
        enet_coef = self.ENet.coef_.ravel()

        self.n_samples, self.n_features = X.shape
        n,m = self.n_samples,self.n_features

        #Get the indexes of the zero coefficient features
        zero_idx = np.where(enet_coef==0)[0]

        if len(zero_idx)<self.n_features:
            if self.fit_intercept:
                init_pen = 1
                m+=1

            #Zou and Hastie 2005 rescaling of ElasticNet coefficients to naive ElasticNet coefficients
            if self.rescale_EN:
                EN_rescale_ctant = 1+((1-self.l1_ratio)/(2*self.C))
                enet_coef = EN_rescale_ctant*enet_coef
                
            beta_variables = cvxpy.Variable(m)
            X_exp = np.c_[np.ones((X.shape[0],1)), X]
            model_prediction = X_exp@beta_variables

            #Loss
            cross_entropy_loss = self.C * (cvxpy.sum(-cvxpy.multiply(model_prediction, y) + cvxpy.logistic(model_prediction))) #Negative log likelihood

            weights = np.maximum(np.abs(enet_coef), self.eps_constant)**-self.gamma
            
            l1_coefs = cvxpy.Parameter(m-init_pen, nonneg=True)
            l1_coefs = weights
            l2_coefs = cvxpy.Parameter(1, nonneg=True)
            l2_coefs = (1 - self.l1_ratio) * 0.5
            l1_penalty = self.nu * cvxpy.norm(cvxpy.multiply(l1_coefs,beta_variables[init_pen:]), 1)
            l2_penalty = l2_coefs * cvxpy.norm(beta_variables[init_pen:],2)**2

            # Constrain that those coefficients that were zero for ENet are zero for AdaNet as well.
            constraints = [beta_variables[idx] == 0 for idx in zero_idx+init_pen]

            # Positive coefficients constraint
            if self.positive:
                constraints.append([b >= 0 for b in beta_variables])

            # --- optimization ---
            problem = cvxpy.Problem(
                objective = cvxpy.Minimize(cross_entropy_loss + l1_penalty + l2_penalty),
                constraints=constraints
            ) 
            
            if not self.force_solver:
                try:
                    if self.AdaNet_solver == 'default':
                        problem.solve(warm_start=True)
                        logging.info(
                            "Default solver used. Parameters max_iter and tol disregarded, consider SCS or ECOS as solver")
                    else:
                        solver_dict = self._cvxpy_solver_options(solver=self.AdaNet_solver)
                        problem.solve(**solver_dict)
                except (ValueError, cvxpy.error.SolverError):
                    if self.printing_solver:
                        logging.warning(
                            'Selected solver failed. Using alternative options. Check solver and solver_stats for more details')
                    solvers = list(set(['SCS','CVXOPT','ECOS']).symmetric_difference(set([self.AdaNet_solver])))
                    for solver in solvers:
                        solver_dict = self._cvxpy_solver_options(solver=solver)
                        try:
                            problem.solve(**solver_dict)
                            if 'optimal' in problem.status:
                                self.AdaNet_solver = solver
                                break
                        except (ValueError, cvxpy.error.SolverError):
                            continue
            else:
                solver_dict = self._cvxpy_solver_options(solver=self.AdaNet_solver)
                problem.solve(**solver_dict)
                
            self.solver_stats = problem.solver_stats
            self.solver_status = problem.status
                
            if self.printing_solver:
                if problem.status in ["infeasible", "unbounded"]:
                    logging.warning('Optimization problem status failure')
                elif problem.status not in ["infeasible", "unbounded"]:                 
                    logging.info("Problem status was {} used {}% of max iterations = {}".format(
                        problem.status,
                        np.round(100*problem.solver_stats.num_iters/self.max_iter, decimals = 1),
                        self.max_iter
                        ))


            beta_sol = beta_variables.value
            beta_sol[np.abs(beta_sol) < self.tol] = 0

            intercept, coef = np.array([beta_sol[0]]), beta_sol[1:]
            coef = np.reshape(coef,(1,self.n_features))

            # Check if constraint violation is less than positive_tol.
            if self.positive and self.positive_tol is not None:
                if not all(c.value(self.positive_tol) for c in constraints):
                    raise ValueError(f"positive_tol is violated. coef is:\n{coef}")
        
        else:
            coef = np.zeros((1,self.n_features))
            if self.printing_solver:
                logging.info('Penalty did not select any nonzero feature: Model is trivial and retuned ENet intercept')
            weights = np.inf*np.ones((1,self.n_features))
            intercept = self.ENet.intercept_
        
        return coef, intercept, enet_coef, weights
  
    
    def _cvxpy_solver_options(self, solver):
        if solver == 'ECOS':
            solver_dict = dict(solver=solver,
                                verbose = self.AdaNet_solver_verbose,
                                warm_start = self.warm_start,
                                max_iters = self.max_iter
                               )
        elif solver == 'SCS':
            solver_dict = dict(solver=solver,
                               max_iters=self.max_iter,
                               verbose = self.AdaNet_solver_verbose,
                               warm_start = self.warm_start
                               )
        elif solver == 'CVXOPT':
            solver_dict = dict(solver=solver,
                        max_iters=self.max_iter,
                        refinement = self.refinement,
                        verbose = self.AdaNet_solver_verbose,
                        warm_start = self.warm_start
                        )
        else:
            solver_dict = dict(solver=solver)

        return solver_dict