import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm,bernoulli
import matplotlib.pyplot as plt

class MLEMAPExamples:
    def __init__(self):
        self.generate_classification_data()
        self.generate_regression_data()

    def generate_classification_data(self):
        true_p=0.7
        self.n_samples_cls=100
        self.X_cls=bernoulli.rvs(true_p,size=self.n_samples_cls)

    def generate_regression_data(self):
        true_params={'beta':2.5,'sigma':1.0}
        self.n_samples_reg=50
        self.X_reg=np.linspace(0,10,self.n_samples_reg)
        noise=np.random.normal(0,true_params['sigma'],self.n_samples_reg)
        self.y_reg=true_params['beta']*self.X_reg+noise

    def classification_mle(self):
        p_mle=np.mean(self.X_cls)
        return p_mle

    def classification_map(self,prior_alpha=10,prior_beta=10):
        successes=np.sum(self.X_cls)
        n=len(self.X_cls)
        p_map=(prior_alpha+successes-1)/(prior_alpha+prior_beta+n-2)
        return p_map

    def regression_neg_log_likelihood(self,params,X,y):
        beta,sigma=params
        mu=beta*X
        return np.sum(norm.logpdf(y,mu,sigma))

    def regression_neg_log_posterior(self, params, X, y,prior_mu,prior_sigma):
        beta,sigma=params
        nll=self.regression_neg_log_likelihood(params,X,y)
        log_prior_beta=-0.5*((beta -prior_mu['beta'])/prior_sigma['beta'])**2
        log_prior_sigma=-2*np.log(sigma)
        return nll-log_prior_beta-log_prior_sigma

    def regression_mle(self):
        initial_params=[1.0,1.0]
        result=minimize(
            self.regression_neg_log_likelihood,
            initial_params,
            args=(self.X_reg,self.y_reg),
            method='Nelder-Mead'
        )
        return result.x
    


