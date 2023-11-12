'''
Author: Marco Barbero
Start Date of implementation: 28th August 2023
Description: Adaptive elastic net model classifier
Version: In this version, the AdaNet loss optimization only includes the active set from
the naive ENet active set. Those features in the non-active hard-thresholded to zero instead
of passing the full dimensionality and a zero constraint to the optimizer.
'''
import sys
import logging 
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
# Suppress the PearsonRConstantInputWarning
warnings.filterwarnings("ignore", category=RuntimeWarning, message=\
                        "An input array is constant; the correlation coefficient is not defined.*")

import pandas as pd
import cvxpy
import numpy as np
from scipy.stats import pointbiserialr
from sklearn.exceptions import ConvergenceWarning
from sklearn.base import MultiOutputMixin,ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted


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
    -SIS_method: Method used to calculate how many features to keep after the sure independence screening. 
        We used a modified version from the original paper (Fan and Lv 2008) based on the point biserial correlation between features and binary outcome.
        As recommended in the paper,the screening method can be'one-less' (n-1), 'log' (n/log(n)), a user-defined integer or None if no screening is desired. 
        If the input dimensionality is too high and 'log' gives a number > number of intances SIS defaults to the number of instances. (Default = 'on-less')
    -AdaNet_solver: What cvxpy solver to use to minimize the AdaNet loss function. By default is the option 'default' that leaves cvxpy teh option to use the most
    appropriate. User can input any solver. If either 'default' or user-defined solver fails the model tries to use either 'ECOS' or 'SCS' to find a 
    suboptimal solution. If none of those works, it reports zero coefficients (trivial solution). (default = 'default')
    -AdaNet_solver_verbose: print out the verbose of the AdaNet solver: cvxpy solver ECOS. (default = False)
    -positive: Positive constraint on coefficients (default = False)
    -Nonzero_tol: Sparseness threshold for small coefficients returned during optimization. It accepts float values,'default' or None. 'default' estimated the threshold
    as ENet tolerance. None does no thresholding. (default = 'default')
    -eps_constant: Small constant to avoid dividing by zero when constructing the adptive weights (default 1e-8).
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
        SIS_method = 'one-less',
        solver_ENet = 'saga',
        AdaNet_solver_verbose = False,
        AdaNet_solver = 'default',
        eps_constant = 1e-8,
        refinement = 10,
        tol=1e-4,
        positive=False,
        nonzero_tol='default',
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
        self.SIS_method = SIS_method #Determines how many features to keep after sure independence screening method
        self.positive = positive
        self.nonzero_tol = nonzero_tol
        self.eps_constant = eps_constant # Small constant to avoid dividing by zero when constructing the adaptive weights
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

        self.classes_ = np.unique(y)

        self.coef_, self.intercept_, self.enet_coef_,self.enet_intercept_, self.weights_ = self._ae(np.asarray(X), np.asarray(y))

        return self

    def predict(self, X):
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict(np.asarray(X))
    
    def predict_proba(self, X): 
        check_is_fitted(self, ["coef_", "intercept_"])
        return super().predict_proba(np.asarray(X))
    
    def predict_proba_ENet(self,X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        linear_pred = X@self.enet_coef_.T+self.enet_intercept_
        exp_pred = np.exp(-linear_pred)
        pos_probas = 1/(1+exp_pred)
        neg_probas = 1-pos_probas.flatten()
        return np.hstack((neg_probas.reshape(-1,1),pos_probas))
    
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

    def sure_independence_screening(self, X, y):
        """Inplementation of Fan and Lv 2008 sure independence screening (SIS) method for binary
        classification.
        In a nutshell, this method reduces the dimensionality of the regression 
        problem before fitting the AdaNet model (also before the initial naive ENet).

        This is specially critical in the p>>n case. Where the assymptotic 
        oracle properties are compromised.

        SIS ensures that the most irrelevant features (in terms of correlation with the outcome)
        are disregarded while keeping all the relevant features with P -> 1. We compute correlations
        based on the point biserial correlation coefficients..

        Uses method specified in the model call to set how many features to keep. As recommended in 
        Fan and Lv 2008,  this can be'one-less' (n-1), 'log' (n/log(n)) or a user-defined integer.
        """
        n,p = X.shape

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        if not np.array_equal(np.unique(y), [0,1]):
            raise ValueError('Labels are not binarized. Please do so before training model')

        if self.SIS_method == 'one-less':
            self.n_out = n-1
        elif self.SIS_method == 'log':
            self.n_out = int(n/np.log(n))
            if self.n_out > n:
                self.n_out = n
        elif isinstance(self.SIS_method, int) or isinstance(self.SIS_method, float):
            self.n_out = self.SIS_method
        elif isinstance(self.SIS_method, str):
            raise ValueError('Wrong SIS method string. Available options are one-less, log or an integer.')

        # Compute Pearson correlation coefficients
        omega = np.array([pointbiserialr(x=y, y=X[:, i])[0] for i in range(X.shape[1])])

        # find the index of the top n_out features
        abs_sorted_indices = np.argsort(-abs(omega))  # Sort in descending order
        original_indices = np.arange(len(omega))

        return  original_indices[abs_sorted_indices][:self.n_out], omega
    
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

        self.n_samples, self.n_features = X.shape

        # Sure independence screening
        if self.SIS_method != None:
            self.idx_out, self.omega = self.sure_independence_screening(X,y)
            X = X[:,self.idx_out] # Select the features that passed the screening
        else:
            self.idx_out = None
            self.omega = None

        # Fit naive ElasticNet
        self.ENet = self.elastic_net().fit(X, y)
        enet_coef = self.ENet.coef_.ravel()

        # Get the indexes of the zero coefficient features
        zero_idx = np.where(enet_coef.ravel()==0)[-1]
        
        # Get the indexes of the NONzero coefficient features
        nonzero_idx = np.array(list(set(np.arange(X.shape[1]))-set(zero_idx)))

        # Select the data on the naive ENet active set 
        X = X[:, nonzero_idx]

        # Shape relevant to the optimization problem
        n,m = X.shape

        if m>0:
            if self.fit_intercept:
                init_pen = 1
                m+=1

            # Zou and Hastie 2005 rescaling of ElasticNet coefficients to naive ElasticNet coefficients
            if self.rescale_EN:
                EN_rescale_ctant = 1+((1-self.l1_ratio)/(2*self.C))
                enet_coef = EN_rescale_ctant*enet_coef
                
            beta_variables = cvxpy.Variable(m)
            X_exp = np.c_[np.ones((n,1)), X]
            model_prediction = X_exp@beta_variables

            # Loss
            cross_entropy_loss = cvxpy.sum(-cvxpy.multiply(model_prediction, y) + cvxpy.logistic(model_prediction)) #Negative log likelihood

            weights = (np.abs(enet_coef[nonzero_idx])+self.eps_constant)**-self.gamma
            
            l1_coefs = cvxpy.Parameter(m-init_pen, nonneg=True)
            l1_coefs = weights
            l2_coefs = cvxpy.Parameter(1, nonneg=True)
            l2_coefs = (1 - self.l1_ratio)/(2*self.C)
            l1_penalty = self.nu * cvxpy.norm(cvxpy.multiply(l1_coefs,beta_variables[init_pen:]), 1)
            l2_penalty = l2_coefs * cvxpy.norm(beta_variables[init_pen:],2)**2

            # Constraints
            constraints = []

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
                        if  self.printing_solver:
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
            
            try:
                if self.nonzero_tol == 'default':
                    self.nonzero_tol = self.tol
                    beta_sol[np.abs(beta_sol) < self.nonzero_tol] = 0
                elif isinstance(self.nonzero_tol,float):
                    beta_sol[np.abs(beta_sol) < self.nonzero_tol] = 0
                else:
                    pass

                intercept, coef_values = np.array([beta_sol[0]]), beta_sol[1:]


                # set coefficient and weights array of the post-SIS input data dimensionality
                coef_SIS = self.replace_values_in_coef_array(
                    values=coef_values,
                    positions=nonzero_idx,
                    out_array_length=len(self.idx_out),
                    fill_value=0.0 
                )
                weights = self.replace_values_in_coef_array(
                    values = weights,
                    positions=nonzero_idx,
                    out_array_length=len(self.idx_out),
                    fill_value=np.inf # ENet non active set features have inf weight (recall w_i = 1/beta_ENet_i)
                )

                # set coefficient and weights array of the input data dimensionality
                coef = self.replace_values_in_coef_array(
                    values=coef_SIS,
                    positions=self.idx_out,
                    out_array_length=self.n_features,
                    fill_value=0.0
                )

                weights = self.replace_values_in_coef_array(
                    values= weights,
                    positions=self.idx_out,
                    out_array_length=self.n_features,
                    fill_value=np.nan # Non-SIS features dont have a weight
                )

            except:
                # if the above gives an error is probably due to a large penalty
                # which makes the loss hard to minimize and the optimization problem being unbounded (no solution found).
                # Set all coefficients to zero. 
                intercept = np.array([0])
                coef = np.zeros((1,self.n_features))

            
            # set ENet coefficient array of the input data dimensionality
            enet_coef = self.replace_values_in_coef_array(
                values = enet_coef,
                positions=self.idx_out,
                out_array_length=self.n_features,
                fill_value=0.0
            )
        
        else:
            enet_coef = np.zeros((1,self.n_features))
            coef = np.zeros((1,self.n_features))
            if self.printing_solver:
                logging.info('Penalty did not select any nonzero feature: Model is trivial and retuned ENet intercept')
            weights = np.inf*np.ones((1,self.n_features))
            intercept = self.ENet.intercept_
        
        return coef, intercept, enet_coef, self.ENet.intercept_, weights
  
    
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
    
    def replace_values_in_coef_array(self, values, positions, out_array_length, fill_value = 0.0):
        out_array = np.full((1,out_array_length),fill_value)
        out_array[0,positions] = values
        return out_array