import pymc3 as pm
import numpy as np

def mcmc_sampler(outcome_model,
                 random_state):
    with outcome_model:
        # draw posterior samples
        step = pm.Metropolis()
        trace = pm.sample(10000, 
                          step=step,
                          #tune=10000,
                          tune=1000,
                          cores=1,
                          chains=1,
                          progressbar=False,
                          return_inferencedata=False,
                          random_seed=random_state)
    return trace


def BLR(D,cov_adj=False,
        y_type='C',
        random_state=2021):
    
    X,T,y = D.X,D.T,D.y
    
    outcome_model = pm.Model()
    with outcome_model:
        
        # priors for unknown parameters
        theta0 = pm.Normal('theta0', mu=0, sigma=100)
        theta1 = pm.Normal('theta1', mu=0, sigma=100)
        
        if y_type == 'C':
            sigma_y = pm.InverseGamma('sigma_y', alpha=0.01, beta=0.01)
        
        if cov_adj:
            alpha = pm.Normal('alpha', mu=0, sigma=100, shape=X.shape[1])
            # likelihood (sampling distribution) of observations
            mu = theta0 + theta1 * T + pm.math.dot(X, alpha)
        else:
            # likelihood (sampling distribution) of observations
            mu = theta0 + theta1 * T
        
        if y_type == 'C':
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y)
        elif y_type == 'B':
            p = pm.Deterministic('p', pm.math.sigmoid(mu))  
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y)

    
    trace = mcmc_sampler(outcome_model,random_state)
    
    return trace


def UIP_Dirichlet(D,summaryDs,
                  cov_adj=False,
                  y_type='C',
                  gammas_ps=False,
                  bal_method='NearMatch',
                  random_state = 2021):
    
    X,T,y = D.X,D.T,D.y
    
    K = len(summaryDs.beta_hat)
    nk_seq = np.array([summaryDs[bal_method][i][0] for i in range(K)])
    bal_seq = np.array([summaryDs[bal_method][i][1] for i in range(K)])
    n,d = X.shape
    
    gammas = nk_seq/n
    gammas[gammas>1] = 1

    if gammas_ps:
        pass
#         tmp = (bal_seq/bal_seq.min())**2
#         gammas = gammas * (tmp**2)
    
    outcome_model = pm.Model()
    with outcome_model:

        # effective sample size
        M = pm.Uniform('M',lower=0,upper=np.minimum(nk_seq.sum(),n))

        # dirichlet distribution for the prior weighting
        pis = pm.Dirichlet("pis", gammas)

        mu1 = 0 
        sigma1 = 0
        for k in range(K):
            mu1 += pis[k]*summaryDs[bal_method][k][2][1,0]
            sigma1 += M*pis[k]/(nk_seq[k]*summaryDs[bal_method][k][2][1,1]**2)
        sigma1 = pm.math.sqrt(1/sigma1)

        # priors for unknown parameters
        theta0 = pm.Normal('theta0', mu=0, sigma=100)
        theta1 = pm.Normal('theta1', mu=mu1, sigma=sigma1)
        
        if y_type == 'C':
            sigma_y = pm.InverseGamma('sigma_y', alpha=0.01, beta=0.01)
        
        if cov_adj:
            alpha = pm.Normal('alpha', mu=0, sigma=100, shape=X.shape[1])
            # likelihood (sampling distribution) of observations
            mu = theta0 + theta1 * T + pm.math.dot(X, alpha)
        else:
            # likelihood (sampling distribution) of observations
            mu = theta0 + theta1 * T

        if y_type == 'C':
            y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma_y, observed=y)
        elif y_type == 'B':
            p = pm.Deterministic('p', pm.math.sigmoid(mu))  
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y)
        
    trace = mcmc_sampler(outcome_model,random_state)
    
    return trace


def BPH(D,n_intervals=5,
        cov_adj=False,
        random_state=2021):
    
    # this function is modified from the tutorial
    # https://docs.pymc.io/en/stable/pymc-examples/examples/survival_analysis/survival_analysis.html
    
    # preprocessing
    y_interval = D.y.copy()
    y_interval = np.append(y_interval[D.T==1],[np.max(y_interval)+1,0])
    interval_bounds = np.quantile(np.unique(y_interval),np.linspace(0,1,n_intervals+1))
    # avoid tied values
    interval_bounds[1:-1] = np.array([0.95*item if item in interval_bounds else item for item in interval_bounds[1:-1]])
    intervals = np.arange(n_intervals)
    interval_length = np.diff(interval_bounds)
    last_period = np.array([np.sum(interval_bounds<=yy)-1 for yy in D.y]).astype(int)
    n = D.y.shape[0]
    
    # create some datasets
    event_mat = np.zeros((n, n_intervals))
    event_mat[np.arange(n), last_period] = D.delta
    exposure_mat = np.greater_equal.outer(D.y, interval_bounds[:-1]) * interval_length
    exposure_mat[np.arange(n), last_period] = D.y - interval_bounds[last_period]
    
    with pm.Model() as outcome_model:

        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape=n_intervals)
        theta1 = pm.Normal("theta1", 0, sigma=100)

        if cov_adj:
            alpha = pm.Normal("alpha", 0, sigma=100, shape=D.X.shape[1])
            lambda_ = pm.Deterministic("lambda_", pm.theano.tensor.outer(pm.math.exp(pm.math.dot(D.X,alpha) + theta1*D.T), 
                                                                     lambda0))
        else:
            lambda_ = pm.Deterministic("lambda_", pm.theano.tensor.outer(pm.math.exp(theta1*D.T),lambda0))
            
        mu = pm.Deterministic("mu", exposure_mat*lambda_)

        obs = pm.Poisson("obs", mu, observed=event_mat)
        
    trace = mcmc_sampler(outcome_model,random_state)
        
    return trace


def BPH_UIP_Dirichlet(D,summaryDs,
                      n_intervals=5,
                      cov_adj=False,
                      gammas_ps=False,
                      bal_method='NearMatch',
                      random_state = 2021):
    
    X,T,y = D.X,D.T,D.y

    K = len(summaryDs.beta_hat)
    nk_seq = np.array([summaryDs[bal_method][i][0] for i in range(K)])
    bal_seq = np.array([summaryDs[bal_method][i][1] for i in range(K)])
    n,d = X.shape

    gammas = nk_seq/n
    gammas[gammas>1] = 1

    if gammas_ps:
        pass
#         tmp = (bal_seq/bal_seq.min())**2
#         gammas = gammas * (tmp**2)

    outcome_model = pm.Model()
    with outcome_model:

        # effective sample size
        M = pm.Uniform('M',lower=0,upper=np.minimum(nk_seq.sum(),n))

        # dirichlet distribution for the prior weighting
        pis = pm.Dirichlet("pis", gammas)

        mu1 = 0 
        sigma1 = 0
        for k in range(K):
            mu1 += pis[k]*summaryDs[bal_method][k][2][0,0]
            sigma1 += M*pis[k]/(nk_seq[k]*summaryDs[bal_method][k][2][0,3]**2)
        sigma1 = pm.math.sqrt(1/sigma1)

        # preprocessing
        y_interval = D.y.copy()
        y_interval = np.append(y_interval[D.T==1],[np.max(y_interval)+1,0])
        interval_bounds = np.quantile(np.unique(y_interval),np.linspace(0,1,n_intervals+1))
        # avoid tied values
        interval_bounds[1:-1] = np.array([0.95*item if item in interval_bounds else item for item in interval_bounds[1:-1]])
        intervals = np.arange(n_intervals)
        interval_length = np.diff(interval_bounds)
        last_period = np.array([np.sum(interval_bounds<=yy)-1 for yy in D.y]).astype(int)

        # create some datasets
        event_mat = np.zeros((D.X.shape[0], n_intervals))
        event_mat[np.arange(D.X.shape[0]), last_period] = D.delta
        exposure_mat = np.greater_equal.outer(D.y, interval_bounds[:-1]) * interval_length
        exposure_mat[np.arange(D.X.shape[0]), last_period] = D.y - interval_bounds[last_period]

        lambda0 = pm.Gamma("lambda0", 0.01, 0.01, shape=n_intervals)
        theta1 = pm.Normal('theta1', mu=mu1, sigma=sigma1)

        if cov_adj:
            alpha = pm.Normal("alpha", 0, sigma=100, shape=D.X.shape[1])
            lambda_ = pm.Deterministic("lambda_", pm.theano.tensor.outer(pm.math.exp(pm.math.dot(D.X,alpha) + theta1*D.T), 
                                                                     lambda0))
        else:
            lambda_ = pm.Deterministic("lambda_", pm.theano.tensor.outer(pm.math.exp(theta1*D.T),lambda0))

        mu = pm.Deterministic("mu", exposure_mat*lambda_)
        obs = pm.Poisson("obs", mu, observed=event_mat)

    trace = mcmc_sampler(outcome_model,random_state)
    
    return trace