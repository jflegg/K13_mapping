import pymc as pm
import numpy as np
from pylab import *
import st_cov_fun
import sys
from utils import *

# ============================Ø====================================================
#  Mean function
# ================================================================================
def m_fun(x, beta_0):
    return pm.gp.zero_fn(x) + beta_0

# ================================================================================
#  Make the model
# ================================================================================
def make_model(id, country, site, lon, lat, year, number_tested, number_with, pf, suitability, population, traveltime):

    first_year = 2000
    # set the space and time tolerances for points to be 'different'
    disttol = 5./6378.
    ttol = 1./12

    if all(np.isnan(pf)):
        with_covariates = 0 # swap out as needed. with_covariates = 0 if NOT using covariates. Otherwise 1. 
    else:
        with_covariates = 1 # swap out as needed. with_covariates = 0 if NOT using covariates. Otherwise 1. 

    # Data locations - stack lon, lat, year
    data_mesh = np.vstack((lon, lat, year)).T

    # Find near spatiotemporal duplicates.
    ui, fi, ti =  Unique_ST_points(disttol, ttol, data_mesh)
    mesh_unique= data_mesh[ui,:]

    # Covariance parameters: spatial differentiability, tlc = temporal limiting correlation, sf = sinusoidal fraction, st = scale in time direction, scale = scale in space direction, inc, ecc  = anisotropy,  amp = overall amplitude
    inc = 0
    ecc = 0
    sf = 0
    sd = .5  
 
     # Skew-normal prior on square root of the partial sill parameter (sigma)
    amp_params = {'mu': .0535, 'tau': 1.79, 'alpha': 3.21}
    log_sigma = pm.SkewNormal('log_sigma',value=1.5,**amp_params)
    sigma = pm.Lambda('sigma', lambda log_sigma = log_sigma: np.exp(log_sigma))
  
    # Skew-normal prior on log-spatial-scale (the range, phi_x)
    scale_params = {'mu': -2.54, 'tau': 1.42, 'alpha': -.015} 
    log_scale = pm.SkewNormal('log_scale',value=-0.5,**scale_params)
    scale = pm.Lambda('scale', lambda log_scale = log_scale: np.exp(log_scale))

    # Exponential prior on the temporal-scale (the range, phi_t)
    time_scale = pm.Exponential('time_scale', 0.1,value=0.1)

    # Uniform prior on the temporal limiting correlation (rho)
    tlc = pm.Uniform('tlc', 0.0, 1.0, value=0.01)

    # mean parameters
    beta_0 = pm.Uninformative('beta_0',value=-9)
    beta_1 = pm.Uninformative('beta_1',value=0.1)
    if with_covariates:
        beta_2 = pm.Uninformative('beta_2',value=0.0)
        beta_3 = pm.Uninformative('beta_3',value=0.0)
        beta_4 = pm.Uninformative('beta_4',value=0.0)
        beta_5 = pm.Uninformative('beta_5',value=0.0)

    # covariance function - modified version of Matern based on Stein
    @pm.deterministic(trace = False)
    def C(sigma=sigma, scale=scale, inc=inc, ecc=ecc, time_scale=time_scale, sd=sd, tlc=tlc, sf=sf):
        return pm.gp.FullRankCovariance(st_cov_fun.my_st, amp=sigma, scale=scale, inc=inc, ecc=ecc,st=time_scale, sd=sd, tlc=tlc, sf = sf)

    # mean function 
    @pm.deterministic(trace = False)
    def M(beta_0 = beta_0):
        return pm.gp.Mean(m_fun, beta_0 = beta_0)

    # establish Gaussian Process submodel to generate S
    S = pm.gp.GPSubmodel('S', M, C, mesh_unique, tally_f=False)

    # expand S.f_eval to all points (not just unique ST points)
    @pm.deterministic
    def f_expanded(fe=S.f_eval, fi=fi):
        return fe[fi]

    # Vinv is the reciprocal of the nugget variate for Z(mesh,t) in the model
    Vinv = pm.Gamma('Vinv', alpha=0.001, beta=0.004, value=0.9)

    @pm.potential
    def constraint1(Vinv = Vinv, upper=5.0):
        if Vinv > upper:                    
            return -inf
        else:
            return 0.0  

   # Set starting values -- need to be careful here to make sure that the indices where number_with = 0 or number_tested=number_with are taken care of, otherwise, stochastic field_plus_nugget's value is outside its support
    start_val = number_with*1.0/number_tested
    start_val[np.where(number_with==0)] = 1.-(0.2)**(1./number_tested[np.where(number_with==0)])
    start_val[np.where(number_with==number_tested)] = (0.2)**(1./number_tested[np.where(number_with==number_tested)])

    ## only store field_plus_nugget below this index -- otherwise the storage kills hdf5!!
    fpn_store = 100 

    # loop over observations and get the field_plus_nuggets and likelihood of observation
    field_plus_nugget = []
    for i in range(len(number_tested)):
        if with_covariates:
            if i < fpn_store:
                field_plus_nugget.append(pm.Normal('field_plus_nugget%s'%i, f_expanded[i]+  beta_1*(year[i]-first_year)+  beta_2*pf[i] + beta_3*suitability[i] + beta_4*log10(1+population[i]) + beta_5*log10(1+traveltime[i]), Vinv, value = pm.logit(start_val[i]), trace=True))
            else:    
                field_plus_nugget.append(pm.Normal('field_plus_nugget%s'%i, f_expanded[i]+  beta_1*(year[i]-first_year)+  beta_2*pf[i] + beta_3*suitability[i] + beta_4*log10(1+population[i]) + beta_5*log10(1+traveltime[i]), Vinv, value = pm.logit(start_val[i]), trace=False))
        else:
            if i < fpn_store:
                field_plus_nugget.append(pm.Normal('field_plus_nugget%s'%i, f_expanded[i]+  beta_1*(year[i]-first_year), Vinv, value = pm.logit(start_val[i]), trace=True))
            else:  
                 field_plus_nugget.append(pm.Normal('field_plus_nugget%s'%i, f_expanded[i]+  beta_1*(year[i]-first_year), Vinv, value = pm.logit(start_val[i]), trace=False))
        # work out field plus nugget
        p = pm.Lambda('p', lambda fpn = field_plus_nugget[-1]: pm.invlogit(fpn), trace = False)
        #  binomial likelihood 
        pos = pm.Binomial('pos', number_tested[i], p, value = number_with[i], observed=True, trace=False)

    return locals()

