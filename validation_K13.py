from model import *
from pylab import *
from numpy import *
import pymc as pm
import matplotlib.pyplot as plt
import map_utils
import tables
import model
import os
import imp
imp.reload(model)
import sys
import pickle
import csv
from pymc.gp import GPEvaluationGibbs
from images_plot_K13 import plot_slices_validate
from utils import *
import copy
import math
import pdb

def validate(data, name, N1, N2, N3, thin, first_year, with_covariates,  continent = "Africa"):

    close('all')
    # ================================================================================
    #   Split the data up into 5 groups and store for later use
    # ================================================================================
    groups = 10  # split the data up into 5 groups to do the validation
    choice = np.ceil(np.random.rand(len(data['lon']))*groups)
    validation_vector = nan*zeros(len(data['lon']))
    within_CI_width = zeros(100)
    
    
    # ================================================================================
    # Save necessary info to reproduce these results and figures to a pickle file
    # ================================================================================
    temp_dict = {'data':data, 'choice': choice, 'N1':N1, 'N2':N2, 'N3':N3, 'name': name, 'first_year': first_year, 'thin': thin, 'with_covariates': with_covariates}
    file = open("validations_dict.pck", "wb")
    pickle.dump(temp_dict, file)
    file.close()

    
    # ================================================================================
    #   For each of the five groups, isolate the validation data set 
    #   and the working data set and reperform the inference
    # ================================================================================
    for i in range(groups):
        close('all')
        # ================================================================================
        # For the ith validation set, find the indexes to use and not to use
        # ================================================================================
        print('validatation %s'%i)
        ind_choice = np.where(choice != (i+1))
        not_ind_choice = np.where(choice == (i+1))

        # ================================================================================
        # Get the data to be used in the validation process
        # ================================================================================
        data_store = copy.deepcopy(data)  # store a copy for accessing original data
        data1 = copy.deepcopy(data)
        dtypes =  list(data1) 
        for datatype in dtypes:
             temp = data[datatype]
             data1[datatype] = temp[ind_choice]
        

        # ================================================================================
        # Figure out what the database file is supposed to be
        # ================================================================================
        hf_name = 'res_db_validation%s'%i+'.hdf5'
        hf_path,hf_basename = os.path.split(hf_name)
        prev_db = None
        if hf_path=='':
            hf_path='./'
        if hf_basename in os.listdir(hf_path):
            prev_db = pm.database.hdf5.load(os.path.join(hf_path,hf_basename))
        if prev_db is not None:  # don't ask, just append to existing database, if it exists already
            ResistanceSampler = pm.MCMC(model.make_model(**data1),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)
        else:
            ResistanceSampler = pm.MCMC(model.make_model(**data1),db='hdf5',dbmode='w', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)


        # ================================================================================
        #   Do the inference 
        # ================================================================================
        # set the step methods
        ResistanceSampler.use_step_method(GPEvaluationGibbs, ResistanceSampler.S, 1/ResistanceSampler.Vinv, ResistanceSampler.field_plus_nugget, ti=ResistanceSampler.ti)
        def isscalar(s):
            return (s.dtype != np.dtype('object')) and (np.alen(s.value)==1) and (s not in ResistanceSampler.field_plus_nugget)
        scalar_stochastics = filter(isscalar, ResistanceSampler.stochastics)
        ResistanceSampler.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, delay=5000, interval=500)
        ResistanceSampler.step_method_dict[ResistanceSampler.log_sigma][0].proposal_sd *= .1
        # do the sampling
        ResistanceSampler.isample(N1, N2, N3) 
        

        # ================================================================================
        # How many saved traces there are:
        # ================================================================================
        n = len(ResistanceSampler.trace('Vinv')[:])
      

        # ================================================================================
        # Plot the parameter traces, in created path folder \Validationi
        # ================================================================================
        mypath = 'validate%s/'%i
        if not os.path.isdir(mypath):
            os.makedirs(mypath) 


        if with_covariates:
            ToPlot = ['Vinv','beta_0','beta_1','beta_2','beta_3', 'beta_4', 'beta_5', 'tlc','log_scale','log_sigma','time_scale']
        else:
            ToPlot = ['Vinv','beta_0','beta_1','tlc','log_scale','log_sigma','time_scale']
        for PLOT in ToPlot:
            print(PLOT)
            close('all')
            figure()
            subplot(1, 2, 1)
            plot(range(len(ResistanceSampler.trace('%s'%PLOT)[:])), ResistanceSampler.trace('%s'%PLOT)[:])
            subplot(1, 2, 2)
            pm.Matplot.histogram(ResistanceSampler.trace('%s'%PLOT)[:], '%s'%PLOT, datarange=(None, None), rows=1, columns=2, num=2, last=True, fontmap={1:10, 2:8, 3:6, 4:5,5:4})
            savefig(mypath+"Validate%s"%i+"_%s.png"%PLOT)



        # ================================================================================
        # Plot the field_plus_nugget traces (only ones that exist!)
        # ================================================================================
        for j in range(len(ResistanceSampler.lon)):
            try: 
                close("all")
                figure(figsize=(10, 6))
                plot(range(len(ResistanceSampler.trace('field_plus_nugget%s'%j)[:])), ResistanceSampler.trace('field_plus_nugget%s'%j)[:])
                savefig(mypath+"Validate%s"%i+"_field_plus_nugget_%s.png"%j)
            except Exception:
                pass


        # ================================================================================
        # Find the model prediction values at the withheld data points
        # not_ind_choice plot of actual versus predicted values at the data locations
        # ================================================================================
        years = data_store['year']
        dplot2 = vstack([data_store['lon'], data_store['lat'], data_store['year']]).T
        if with_covariates:
            pf = data_store['pf']
            suitability = data_store['suitability']
            population = data_store['population']
            traveltime = data_store['traveltime']


        temp = zeros([len(not_ind_choice[0]), int(math.ceil(float(n)/thin))])
        count = 0
        # Get E[v] and over the entire posterior
        for j in range(0,n, thin):
            # Reset all variables to their values at frame j of the trace
            ResistanceSampler.remember(0,j)
            # Evaluate the observed mean
            Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2)
            Vsurf_temp += 1.0/ResistanceSampler.Vinv.value    
            if (all(Vsurf_temp>=0)):
                if with_covariates:
                    Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(years-2000) + ResistanceSampler.trace('beta_2')[i]*pf + ResistanceSampler.trace('beta_3')[i]*suitability + ResistanceSampler.trace('beta_4')[i]*log10(1+population) + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime)
                else:
                    Msurf_temp +=  ResistanceSampler.trace('beta_1')[j]*(years-2000) 
                freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
                temp[:,count] = freq[not_ind_choice]
                count += 1
        

        # ================================================================================
        # Plot the actual versus precticted values, for those data that were NOT used
        # ================================================================================
        Med = 100.*median(temp[:,range(count)],axis=1)
        for kk in range(1,101):
            Qtemp_lo = percentile(temp[:,:], 50-1.0*kk/2, axis=1)
            Qtemp_hi = percentile(temp[:,range(count)], 50+1.0*kk/2, axis=1)
            within_CI_width[kk-1] = within_CI_width[kk-1] + len(where((data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]>=Qtemp_lo)& (data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]<=Qtemp_hi) )[0])
        
        # plot which points are in 50% CI and those that aren't
        m1= len(years[not_ind_choice] )
        close('all')
        figure()
        outliers = []
        R_1 = percentile(temp, 25, axis = 1)
        R_2 = percentile(temp, 75, axis = 1)
        resis = data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]

        for ii in range(m1):
            x = ii
            ymin = R_1[ii]
            ymax = R_2[ii]
            axvline(x=x, ymin=ymin, ymax=ymax, color = 'r', linewidth = 1)
            dv = resis[ii]/100
            if (dv < ymin) | (dv > ymax):
                outliers += [ii]    

        plot(range(m1), resis ,'k.',markersize=5)
        plot(outliers, resis[outliers] ,'b.',markersize=5)
        title('')
        xlabel('Data point number')
        ylabel('CI and actual data point value')
        savefig(mypath+'ConfInts_points_%s.pdf'%i)


        close('all')
        figure()
        resis = data['number_with'][not_ind_choice]*(1.0/data['number_tested'][not_ind_choice])*100.
        plot(resis,Med,'k.',markersize=5)
        plot(range(100), range(100))
        axis(np.array([0,110,0,110]))
        title('Actual versus predicted')
        xlabel('Actual resistance (%)')
        ylabel('Predicted resistance (%)')
        savefig(mypath+'Actual_vs_predicted_%s'%i+'.png')

        # ================================================================================
        # Print the correlation between predicted and actual values
        # ================================================================================
        try:
            correlation = np.corrcoef(Med, resis)[0,1]
            print(correlation)
        except:
            print('could not calculate correlation coefficient')
        validation_vector[not_ind_choice] = Med
        print(Med)
        print(resis)
        print(ind_choice)
        print(not_ind_choice)

        # ================================================================================
        # Do the plotting of slices
        # ================================================================================
        plot_slices_validate(name,first_year, thin,i,mypath, data1, with_covariates, continent)


    # ================================================================================
    # Plot the entire actual versus predicted plot
    # ================================================================================
    print("Plotting entire validation plot")
    close('all')
    figure()
    resis = data['number_with']*(1./data['number_tested'])*100.
    plot(resis,validation_vector,'k.',markersize=5)
    plot(range(100), range(100))
    axis(np.array([0,110,0,110]))
    title('Validation - Actual versus predicted')
    xlabel('Actual resistance (%)')
    ylabel('Predicted resistance (%)')
    savefig('Actual_vs_predicted_ALL.png')

    # ================================================================================
    # Print correlation coeffiection for the entire actual versus precticted values
    # ================================================================================
    try:
        correlation = np.corrcoef(validation_vector, resis)[0,1]
        print(correlation)
    except:
        print('could not calculate correlation coefficient')
    print(validation_vector)
    print(resis)

 
    # ================================================================================
    # Save predicted values (median) and observed frequencies
    # ================================================================================
    output = vstack([data['lon'], data['lat'], data['year'], data['number_tested'], data['number_with'], resis, validation_vector]).T
    savetxt('validation_output.csv',output, fmt='%.5f',delimiter = ',',header = "lon, lat, year, number_tested, number_with, resistance, validation_vector", comments = '')
    output = vstack([within_CI_width]).T
    savetxt('validation_output2.csv',output, fmt='%.5f',delimiter = ',',header = "within_CI_width", comments = '')


    # ================================================================================
    # Print the ME and MAE values to screen
    # ================================================================================
    ME = np.mean(validation_vector-resis)
    MAE = np.mean(np.abs(validation_vector-resis))
    print("ME = %s"%ME)
    print("MAE = %s"%MAE)

    return locals()



def validate_single_group(data, name, N1, N2, N3, thin, first_year, choice, i, with_covariates,  continent = "Africa"):

    close('all')
    # ================================================================================
    # For the ith validation set, find the indexes to use and not to use
    # ================================================================================
    print('validatation %s'%i)
    ind_choice = np.where(choice != (i+1))
    not_ind_choice = np.where(choice == (i+1))

    # ================================================================================
    # Get the data to be used in the validation process
    # ================================================================================
    data_store = copy.deepcopy(data)  # store a copy for accessing original data
    data1 = copy.deepcopy(data)
    dtypes =  list(data1) 
    for datatype in dtypes:
        temp = data[datatype]
        data1[datatype] = temp[ind_choice]
        

    # ================================================================================
    # Figure out what the database file is supposed to be
    # ================================================================================
    hf_name = 'res_db_validation%s'%i+'.hdf5'
    hf_path,hf_basename = os.path.split(hf_name)
    prev_db = None
    if hf_path=='':
        hf_path='./'
    if hf_basename in os.listdir(hf_path):
        prev_db = pm.database.hdf5.load(os.path.join(hf_path,hf_basename))
    if prev_db is not None:  # don't ask, just append to existing database, if it exists already
        ResistanceSampler = pm.MCMC(model.make_model(**data1),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)
    else:
        ResistanceSampler = pm.MCMC(model.make_model(**data1),db='hdf5',dbmode='w', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)


    # ================================================================================
    #   Do the inference 
    # ================================================================================
    # set the step methods
    ResistanceSampler.use_step_method(GPEvaluationGibbs, ResistanceSampler.S, 1/ResistanceSampler.Vinv, ResistanceSampler.field_plus_nugget, ti=ResistanceSampler.ti)
    def isscalar(s):
        return (s.dtype != np.dtype('object')) and (np.alen(s.value)==1) and (s not in ResistanceSampler.field_plus_nugget)
    scalar_stochastics = filter(isscalar, ResistanceSampler.stochastics)
    ResistanceSampler.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, delay=5000, interval=500)
    ResistanceSampler.step_method_dict[ResistanceSampler.log_sigma][0].proposal_sd *= .1
    # do the sampling
    ResistanceSampler.isample(N1, N2, N3) 
        

    # ================================================================================
    # How many saved traces there are:
    # ================================================================================
    n = len(ResistanceSampler.trace('Vinv')[:])
      

    # ================================================================================
    # Plot the parameter traces, in created path folder \Validationi
    # ================================================================================
    mypath = 'validate%s/'%i
    if not os.path.isdir(mypath):
        os.makedirs(mypath) 


    if with_covariates:
        ToPlot = ['Vinv','beta_0','beta_1','beta_2','beta_3','beta_4','beta_5','tlc','log_scale','log_sigma','time_scale']
    else:
        ToPlot = ['Vinv','beta_0','beta_1','tlc','log_scale','log_sigma','time_scale']
    for PLOT in ToPlot:
        print(PLOT)
        close('all')
        figure()
        subplot(1, 2, 1)
        plot(range(len(ResistanceSampler.trace('%s'%PLOT)[:])), ResistanceSampler.trace('%s'%PLOT)[:])
        subplot(1, 2, 2)
        pm.Matplot.histogram(ResistanceSampler.trace('%s'%PLOT)[:], '%s'%PLOT, datarange=(None, None), rows=1, columns=2, num=2, last=True, fontmap={1:10, 2:8, 3:6, 4:5,5:4})
        savefig(mypath+"Validate%s"%i+"_%s.png"%PLOT)



    # ================================================================================
    # Plot the field_plus_nugget traces (only ones that exist!)
    # ================================================================================
    for j in range(len(ResistanceSampler.lon)):
        try: 
            close("all")
            figure(figsize=(10, 6))
            plot(range(len(ResistanceSampler.trace('field_plus_nugget%s'%j)[:])), ResistanceSampler.trace('field_plus_nugget%s'%j)[:])
            savefig(mypath+"Validate%s"%i+"_field_plus_nugget_%s.png"%j)
        except Exception:
            pass


    # ================================================================================
    # Find the model prediction values at the withheld data points
    # not_ind_choice plot of actual versus predicted values at the data locations
    # ================================================================================
    years = data_store['year']
    dplot2 = vstack([data_store['lon'], data_store['lat'], data_store['year']]).T
    if with_covariates:
        pf = data_store['pf']
        suitability = data_store['suitability']
        population = data_store['population']
        traveltime = data_store['traveltime']

    temp = zeros([len(not_ind_choice[0]), int(math.ceil(float(n)/thin))])
    count = 0
    # Get E[v] and over the entire posterior
    for j in range(0,n, thin):
        # Reset all variables to their values at frame j of the trace
        ResistanceSampler.remember(0,j)
        # Evaluate the observed mean
        Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2)
        Vsurf_temp += 1.0/ResistanceSampler.Vinv.value    
        if (all(Vsurf_temp>=0)):
            if with_covariates:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(years-2000) + ResistanceSampler.trace('beta_2')[i]*pf + ResistanceSampler.trace('beta_3')[i]*suitability + ResistanceSampler.trace('beta_4')[i]*log10(1+population) + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime)
            else:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[j]*(years-2000) 
            freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
            temp[:,count] = freq[not_ind_choice]
            count += 1
        

    # ================================================================================
    # Plot the actual versus precticted values, for those data that were NOT used
    # ================================================================================
    Med = 100.*median(temp[:,range(count)],axis=1)
    # plot which points are in 50% CI and those that aren't
    m1= len(years[not_ind_choice] )
    close('all')
    figure()
    outliers = []
    R_1 = percentile(temp, 25, axis = 1)
    R_2 = percentile(temp, 75, axis = 1)
    resis = data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]

    for ii in range(m1):
        x = ii
        ymin = R_1[ii]
        ymax = R_2[ii]
        axvline(x=x, ymin=ymin, ymax=ymax, color = 'r', linewidth = 1)
        dv = resis[ii]/100
        if (dv < ymin) | (dv > ymax):
            outliers += [ii]    

    plot(range(m1), resis ,'k.',markersize=5)
    plot(outliers, resis[outliers] ,'b.',markersize=5)
    title('')
    xlabel('Data point number')
    ylabel('CI and actual data point value')
    savefig(mypath+'ConfInts_points_%s.pdf'%i)


    close('all')
    figure()
    resis = data['number_with'][not_ind_choice]*(1.0/data['number_tested'][not_ind_choice])*100.
    plot(resis,Med,'k.',markersize=5)
    plot(range(100), range(100))
    axis(np.array([0,110,0,110]))
    title('Actual versus predicted')
    xlabel('Actual resistance (%)')
    ylabel('Predicted resistance (%)')
    savefig(mypath+'Actual_vs_predicted_%s'%i+'.png')


    # ================================================================================
    # Print the correlation between predicted and actual values
    # ================================================================================
    try:
        correlation = np.corrcoef(Med, resis)[0,1]
        print(correlation)
    except:
        print('could not calculate correlation coefficient')
    print(Med)
    print(resis)
    print(ind_choice)
    print(not_ind_choice)


    # ================================================================================
    # Do the plotting of slices
    # ================================================================================
    plot_slices_validate(name,first_year, thin,i,mypath, data1, with_covariates, continent)


    return locals()




# ================================================================================
# not "redo_from_saved_validate" has not been checked to work!
# ================================================================================
def redo_from_saved_validate(data, name, choice, thin, first_year, with_covariates, continent = "Africa"): 

    # ================================================================================
    #   Split the data up into 5 groups and store for later use
    # ================================================================================
    groups = int(choice.max())      # split the data up into 5 groups to do the validation
    validation_vector = nan*zeros(len(data['lon']))
    within_CI_width = zeros(100)

    # ================================================================================
    #   For each of the five groups, isolate the validation data set 
    #   and the working data set and RELOAD the inference results
    # ================================================================================
    for i in range(groups):
        # ================================================================================
        # For the ith validation set, find the indexes to use and not to use
        # ================================================================================
        print('validatation %s'%i)
        ind_choice = np.where(choice != (i+1))
        not_ind_choice = np.where(choice == (i+1))

        # ================================================================================
        # Get the data to be used in the validation process
        # ================================================================================
        data1 = copy.deepcopy(data)
        dtypes =  list(data1) 
        for datatype in dtypes:
             temp = data[datatype]
             data1[datatype] = temp[ind_choice]

        # ================================================================================
        # Reload the database
        # ================================================================================
        hf_name = 'res_db_validation%s'%i+'.hdf5'
        prev_db = pm.database.hdf5.load(hf_name)
        ResistanceSampler = pm.MCMC(model.make_model(**data1),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)

        # ================================================================================
        # How many saved traces there are:
        # ================================================================================
        n = len(ResistanceSampler.trace('Vinv')[:])

        mypath = 'validate%s/'%i
        if not os.path.isdir(mypath):
            os.makedirs(mypath) 

        # ================================================================================
        # Find the model prediction values at the withheld data points
        # not_ind_choice plot of actual versus predicted values at the data locations
        # ================================================================================
        years = data['year'][not_ind_choice]
        dplot2 = vstack([data['lon'][not_ind_choice], data['lat'][not_ind_choice], data['year'][not_ind_choice]]).T
        if with_covariates:
            pf = data['pf'][not_ind_choice]
            suitability = data['suitability'][not_ind_choice]
            population = data['population'][not_ind_choice]
            traveltime = data['traveltime'][not_ind_choice]

        temp = zeros([dplot2.shape[0], int(math.ceil(float(n)/thin))])
        count = 0
        # Get E[v] and over the entire posterior
        for j in range(0,n, thin):
            # Reset all variables to their values at frame j of the trace
            ResistanceSampler.remember(0,j)
            # Evaluate the observed mean
            Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2)
            Vsurf_temp += 1.0/ResistanceSampler.Vinv.value    
            if (all(Vsurf_temp>=0)):
                if with_covariates:
                    Msurf_temp +=  ResistanceSampler.trace('beta_1')[j]*(years-2000) + ResistanceSampler.trace('beta_2')[j]*pf + ResistanceSampler.trace('beta_3')[i]*suitability + ResistanceSampler.trace('beta_4')[i]*log10(1+population)  + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime)
                else:
                    Msurf_temp +=  ResistanceSampler.trace('beta_1')[j]*(years-2000) 
                freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
                temp[:,count] = freq
                count += 1
        

        # ================================================================================
        # Plot the actual versus precticted values, for those data that were NOT used
        # ================================================================================
        Med = 100.*median(temp[:,range(count)],axis=1)
        for kk in range(1,101):
            Qtemp_lo = percentile(temp[:,range(count)], 50-1.0*kk/2, axis=1)
            Qtemp_hi = percentile(temp[:,range(count)], 50+1.0*kk/2, axis=1)
            within_CI_width[kk-1] = within_CI_width[kk-1] + len(where((data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]>=Qtemp_lo)& (data['number_with'][not_ind_choice]*1.0/data['number_tested'][not_ind_choice]<=Qtemp_hi) )[0])

        close('all')
        figure()
        Med = 100.*median(temp[:,range(count)],axis=1)
        resis = data['number_with'][not_ind_choice]*(1.0/data['number_tested'][not_ind_choice])*100.
        plot(resis,Med,'k.',markersize=5)
        plot(range(100), range(100))
        axis(np.array([0,110,0,110]))
        title('Actual versus predicted')
        xlabel('Actual resistance (%)')
        ylabel('Predicted resistance (%)')
        savefig(mypath+'Actual_vs_predicted_%s'%i+'.png')

        # ================================================================================
        # Print the correlation between predicted and actual values
        # ================================================================================
        try:
            correlation = np.corrcoef(Med, resis)[0,1]
            print(correlation)
        except:
            print('could not calculate correlation coefficient')
        validation_vector[not_ind_choice] = Med

        # ================================================================================
        # Do the plotting of slices
        # ================================================================================
        plot_slices_validate(name,first_year, thin,i,mypath, data1, with_covariates, continent)

        # ================================================================================
        # Plot the parameter traces, in created path folder \Validationi
        # ================================================================================
        if with_covariates:
            ToPlot = ['Vinv','beta_0','beta_1','beta_2','beta_3','beta_4','beta_5','tlc','log_scale','log_sigma','time_scale']
        else:
            ToPlot = ['Vinv','beta_0','beta_1','tlc','log_scale','log_sigma','time_scale']
        for PLOT in ToPlot:
            print(PLOT)
            close('all')
            figure()
            subplot(1, 2, 1)
            plot(range(len(ResistanceSampler.trace('%s'%PLOT)[:])), ResistanceSampler.trace('%s'%PLOT)[:])
            subplot(1, 2, 2)
            pm.Matplot.histogram(ResistanceSampler.trace('%s'%PLOT)[:], '%s'%PLOT, datarange=(None, None), rows=1, columns=2, num=2, last=True, fontmap={1:10, 2:8, 3:6, 4:5,5:4})
            savefig(mypath+"Validate%s"%i+"_%s.png"%PLOT)


        # ================================================================================
        # Plot the field_plus_nugget traces (only ones that exist!)
        # ================================================================================
        for j in range(len(ResistanceSampler.lon)):
            try: 
                close("all")
                figure(figsize=(10, 6))
                plot(range(len(ResistanceSampler.trace('field_plus_nugget%s'%j)[:])), ResistanceSampler.trace('field_plus_nugget%s'%j)[:])
                savefig(mypath+"Validate%s"%i+"_field_plus_nugget_%s.png"%j)
            except Exception:
                pass
                        

    # ================================================================================
    # Plot the entire actual versus predicted plot
    # ================================================================================
    print("Plotting entire validation plot")
    close('all')
    figure()
    resis = data['number_with']*(1./data['number_tested'])*100.
    plot(resis,validation_vector,'k.',markersize=5)
    plot(range(100), range(100))
    axis(np.array([0,110,0,110]))
    title('Validation - Actual versus predicted')
    xlabel('Actual resistance (%)')
    ylabel('Predicted resistance (%)')
    savefig('Actual_vs_predicted_ALL.png')

    # ================================================================================
    # Print correlation coeffiection for the entire actual versus precticted values
    # ================================================================================
    try:
        correlation = np.corrcoef(validation_vector, resis)[0,1]
        print(correlation)
    except:
        print('could not calculate correlation coefficient')

    # ================================================================================
    # Save predicted values (median) and observed frequencies
    # ================================================================================

    output = vstack([data['lon'], data['lat'], data['year'], data['number_tested'], data['number_with'], resis, validation_vector]).T
    fid = open('validation_output.csv', 'w')
    fid.write("lon, lat, year, number tested, number with, resistance, validation_vector\n")
    savetxt(fid,output, fmt='%.5f',delimiter = ',')
    fid.close()

    output = vstack([within_CI_width]).T
    fid = open('validation_output2.csv', 'w')
    fid.write("within_CI_width\n")
    savetxt(fid,output, fmt='%.5f',delimiter = ',')
    fid.close()



    # ================================================================================
    # Print the ME and MAE values to screen
    # ================================================================================
    ME = np.mean(validation_vector-resis)
    MAE = np.mean(np.abs(validation_vector-resis))
    print("ME = %s"%ME)
    print("MAE = %s"%MAE)
    
    return locals()

