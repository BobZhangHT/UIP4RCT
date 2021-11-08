import pandas as pd
import numpy as np
from rpy2 import robjects as ro
from rpy2.robjects import Formula, FloatVector
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri

numpy2ri.activate()
pandas2ri.activate()

r_weightit = importr("WeightIt")
r_matchit = importr("MatchIt")
r_stats = importr('stats')
r_cobalt = importr('cobalt')
r_sandwich = importr("sandwich")
r_lmtest = importr("lmtest")
r_survival = importr('survival')

def matchit_wrapper(y,t,x,
                    cov_adj=False,
                    y_type='C',
                    method='full',
                    estimand='ATE',
                    distance='glm',
                    replace=False,
                   ):

    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['t','y']
    formula = columns[-2]+'~'+'+'.join(columns[:-2])

    kwargs = {
        'formula': Formula(formula),
        "data": pd.DataFrame(np.hstack([x, t.reshape(-1,1),y.reshape(-1,1)]),
                                        columns=columns),
        'distance': distance,
        'method': method,
        'replace': replace,
        'estimand': estimand
    }
    

    m_out = r_matchit.matchit(**kwargs)
    bal_out = r_cobalt.bal_tab(m_out, un = True)
    
    if replace:
        m_data = r_matchit.get_matches(m_out)
    else:
        m_data = r_matchit.match_data(m_out)
    
    if y_type == 'C':
        if cov_adj:
            fit = r_stats.lm(Formula('y~'+'+'.join(['t']+columns[:-2])),
                         data=m_data,weights=m_data['weights'])
        else:
            fit = r_stats.lm(Formula('y~t'),data=m_data,weights=m_data['weights'])
            
    elif y_type == 'B':
        if cov_adj:
            fit = r_stats.glm(Formula('y~'+'+'.join(['t']+columns[:-2])),
                              family='binomial',
                              data=m_data,weights=m_data['weights'])
        else:
            fit = r_stats.glm(Formula('y~t'),family='binomial',
                              data=m_data,weights=m_data['weights'])
        
    summary_fit = r_lmtest.coeftest(fit, vcov_= r_sandwich.vcovCL, cluster = Formula('~subclass'))
    
    return m_data, summary_fit, bal_out



def surv_matchit_wrapper(y,t,delta,x,
                    cov_adj=False,
                    method='full',
                    estimand='ATE',
                    distance='glm',
                    replace=False
                   ):

    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['t','status','y']
    formula = 't~'+'+'.join(columns[:-3])
    
    kwargs = {  'formula': Formula(formula),
            "data": pd.DataFrame(np.hstack([x, t.reshape(-1,1),delta.reshape(-1,1),y.reshape(-1,1)]),
                                            columns=columns),
            'distance': distance,
            'method': method,
            'replace': replace,
            'estimand': estimand
         }

    m_out = r_matchit.matchit(**kwargs)
    bal_out = r_cobalt.bal_tab(m_out, un = True)

    if replace:
        m_data = r_matchit.get_matches(m_out)
    else:
        m_data = r_matchit.match_data(m_out)

    if cov_adj:
        fit = r_survival.coxph(Formula('Surv(y,status)~t+'+'+'.join(columns[:-3])),
                           robust=True, data=m_data,
                           weights=FloatVector(m_data['weights']), 
                           cluster=m_data['subclass'])
    else:
        fit = r_survival.coxph(Formula('Surv(y,status)~t'),
                           robust=True, data=m_data,
                           weights=FloatVector(m_data['weights']), 
                           cluster=m_data['subclass'])

    summary_fit = r_survival.summary_coxph(fit)[6]
    
    return m_data, summary_fit, bal_out


def weightit_wrapper(y,t,x,
                    cov_adj=False,
                    y_type='C',
                    method='ps',
                    estimand='ATE',
                    link='logit'):

    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['t','y']
    formula = columns[-2]+'~'+'+'.join(columns[:-2])

    w_data = pd.DataFrame(np.hstack([x, t.reshape(-1,1),y.reshape(-1,1)]),
                                        columns=columns)

    kwargs = {
        'formula': Formula(formula),
        "data": w_data,
        'link': link,
        'method': method,
        'estimand': estimand
    }


    w_out = r_weightit.weightit(**kwargs)
    bal_out = r_cobalt.bal_tab(w_out, un = True)
    w_data['weights'] = w_out[0]
    
    if y_type == 'C':
        if cov_adj:
            fit = r_stats.lm(Formula('y~'+'+'.join(['t']+columns[:-2])),
                         data=w_data,weights=w_out[0])
        else:
            fit = r_stats.lm(Formula('y~t'),data=w_data,weights=w_out[0])
            
    elif y_type == 'B':
        if cov_adj:
            fit = r_stats.glm(Formula('y~'+'+'.join(['t']+columns[:-2])),
                              family='binomial',
                              data=w_data,weights=w_out[0])
        else:
            fit = r_stats.glm(Formula('y~t'),family='binomial',
                              data=w_data,weights=w_out[0])
        
    summary_fit = r_lmtest.coeftest(fit, vcov_=r_sandwich.sandwich)
    
    return w_data, summary_fit, bal_out


def surv_weightit_wrapper(y,t,delta,x,
                    cov_adj=False,
                    method='ps',
                    estimand='ATE',
                    link='logit'
                   ):

    columns = ['x'+str(i+1) for i in range(x.shape[1])] + ['t','status','y']
    formula = 't~'+'+'.join(columns[:-3])
    
    w_data = pd.DataFrame(np.hstack([x, t.reshape(-1,1),delta.reshape(-1,1),y.reshape(-1,1)]),
                                            columns=columns)
    kwargs = {
        'formula': Formula(formula),
        "data": w_data,
        'link': link,
        'method': method,
        'estimand': estimand
    }

    w_out = r_weightit.weightit(**kwargs)
    bal_out = r_cobalt.bal_tab(w_out, un = True)
    w_data['weights'] = w_out[0]

    if cov_adj:
        fit = r_survival.coxph(Formula('Surv(y,status)~t+'+'+'.join(columns[:-3])),
                           robust=True, data=w_data,
                           weights=FloatVector(w_data['weights']))
    else:
        fit = r_survival.coxph(Formula('Surv(y,status)~t'),
                           robust=True, data=w_data,
                           weights=FloatVector(w_data['weights']))

    summary_fit = r_survival.summary_coxph(fit)[6]
    
    return w_data, summary_fit, bal_out