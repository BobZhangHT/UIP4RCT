import numpy as np
from scipy.optimize import minimize,dual_annealing,brute,fmin

# data generation
def sigmoid(x):
    return np.exp(x)/(1+np.exp(x))

def y_gen(X,theta,beta,
          y_type='C',
          sig_err=0.5):
    n, d = X.shape
    # generate propensity scores
    Xb = np.dot(X,beta)
    ps = sigmoid(Xb)
    # generate allocation
    T = np.random.binomial(1,ps,ps.shape[0])
    # generate response
    if y_type == 'C':
        y = theta[0]+theta[1]*T+X.dot(theta[2:])+sig_err*np.random.randn(n)
    elif y_type == 'B':
        z = theta[0]+theta[1]*T+X.dot(theta[2:])
        y = np.random.binomial(1,sigmoid(z),z.shape[0])
    return T, y, ps

def cen_fun(theta_c,e_t,cen_ratio=0.1):
    c_t = np.random.uniform(low=0,high=theta_c,size=e_t.shape[0])
    delta = np.array(c_t >= e_t,dtype=int)
    return np.abs(cen_ratio-(1-delta).mean())

def surv_y_gen(X,theta,beta,theta_c,
               alpha=1,v=2):
    
    n, d = X.shape
    # generate propensity scores
    Xb = np.dot(X,beta)
    ps = sigmoid(Xb)
    # generate allocation
    T = np.random.binomial(1,ps,ps.shape[0])
    # xTb
    z = theta[0]*T+X.dot(theta[1:])
    # survival prob
    S = np.random.rand(n)
    # event time
    e_t = ((-np.log(S)/np.exp(z))*(v**alpha))**(1/alpha)
    # generate the censored time
    c_t = np.random.uniform(low=0,high=theta_c,size=n)
    delta = np.array(c_t >= e_t,dtype=bool)
    y = np.minimum(c_t,e_t)
    
    return T, delta, y, e_t, c_t, ps