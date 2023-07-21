from pylab import *
from numpy import *
import pymc as pm
from pymc.gp import GPEvaluationGibbs
import model
import imp
imp.reload(model)
import sys
import pickle
from images_plot_K13 import plot_slices, plot_from_raster
from validation_K13 import *
from utils import *


# ================================================================================
#  MCMC run
# ================================================================================
def mcmc(with_covariates):
  # ================================================================================
  #  Load the data and convert from record array to dictionary
  # ================================================================================
  name = 'K13_June_7_2023_Mekong_Plus_Mutations_for_pymc_mod'
  data = csv2rec(name+'.csv') 
  data = dict([(k,data[k]) for k in data.dtype.names])


  # ================================================================================
  #   Transform from degrees to radians
  # ================================================================================
  data['lat'], data['lon'] = convert_coords(data['lat'],data['lon'], 'degrees', 'radians')


  # ================================================================================
  # Get the covariates needed: pf and suitability and travel time
  # ================================================================================
  # stack the values of lon and lat and year together
  if with_covariates:
    data_mesh = np.vstack((data['lon'],data['lat'], data['year'])).T
    covariates = getCovariatesForLocationsK13maps(data_mesh)
    data['pf'] = covariates[0]
    data['suitability'] = covariates[1]
    data['population'] = covariates[2]
    data['traveltime'] = covariates[3]
  else:
    data['pf'] = np.full(len(data['lon']), np.nan) 
    data['suitability'] = np.full(len(data['lon']), np.nan) 
    data['population'] = np.full(len(data['lon']), np.nan) 
    data['traveltime'] = np.full(len(data['lon']), np.nan) 

  # ================================================================================
  # Figure out what the database file is supposed to be
  # ================================================================================
  hf_name = 'res_db.hdf5'
  hf_path,hf_basename = os.path.split(hf_name)
  prev_db = None
  if hf_path=='':
     hf_path='./'
  if hf_basename in os.listdir(hf_path):
     rm_q = input('\nDatabase file %s already exists in path %s. Do you want to continue sampling? [yes or no] '%(hf_basename, hf_path))
     if rm_q.strip() in ['y','YES','Yes','yes','Y']:
         prev_db = pm.database.hdf5.load(os.path.join(hf_path,hf_basename))
     elif rm_q.strip() in ['n','NO','No','no','N']:
         rm_q = input('\nDo you want me to remove the file and start fresh? [yes or no] ')
         if rm_q.strip() in ['y','YES','Yes','yes','Y']:
             print('\nExcellent.')
             os.remove(hf_name)
         elif rm_q.strip() in ['n','NO','No','no','N']:
             raise OSError("\nOK, I'm leaving it to you to sort yourself out.")
         else:
             raise OSError("\nI don't know what you are trying to say.Move, rename or delete the database to continue.")



  # ================================================================================
  #  Create model     
  # ================================================================================
  if prev_db is not None:
     ResistanceSampler = pm.MCMC(model.make_model(**data),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)
  else:
     ResistanceSampler = pm.MCMC(model.make_model(**data),db='hdf5',dbmode='w', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)

  # ================================================================================
  #   Do the inference 
  # ================================================================================

  N1 = 1000000
  N2 = 500000
  N3 = 100
  thin = 50
  first_year = 2000

  

  # ================================================================================
  # Save necessary info to reproduce plotting results and figures to a pickle file
  # ================================================================================
  Replot_dict = {'data':data, 'N1':N1, 'N2':N2, 'N3':N3, 'thin':thin, 'first_year':first_year, 'name': name, 'with_covariates': with_covariates}
  file = open("Replot_dict.pck", "wb")
  pickle.dump(Replot_dict, file)
  file.close()


  # set the step methods
  ResistanceSampler.use_step_method(GPEvaluationGibbs, ResistanceSampler.S, 1/ResistanceSampler.Vinv, ResistanceSampler.field_plus_nugget, ti=ResistanceSampler.ti)
  def isscalar(s):
     return (s.dtype != np.dtype('object')) and (np.alen(s.value)==1) and (s not in ResistanceSampler.field_plus_nugget)
  scalar_stochastics = filter(isscalar, ResistanceSampler.stochastics)
  ResistanceSampler.use_step_method(pm.gp.GPParentAdaptiveMetropolis, scalar_stochastics, delay=5000, interval=500)
  ResistanceSampler.step_method_dict[ResistanceSampler.log_sigma][0].proposal_sd *= .1

  # do the sampling
  ResistanceSampler.isample(N1, N2, N3, verbose = 0)  # put verbose = 1 to get all the output.... 


  # ================================================================================
  #   Do the plotting of slices, traces, autocorrlations etc 
  # ================================================================================
  continent = "Mekong"
  plot_slices(data, name,first_year, thin, with_covariates, continent)


  # ================================================================================
  #   Do the validataions
  # ================================================================================
  first_year = 2019
  Validate = validate(data,name, N1, N2, N3, thin, first_year, with_covariates)


# ================================================================================
#  RePlotCommands
# ================================================================================
def RePlotCommands():
  # ================================================================================
  #  Load the data from the pickle file
  # ================================================================================
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']
  with_covariates = Replot_dict['with_covariates']

  # ================================================================================
  #   Do plotting of slices, traces, autocorrlations etc, from the save databases 
  # ================================================================================
  plot_slices(data, name,first_year, thin, with_covariates)



# ================================================================================
#  RePlotCommands_from_raster
# ================================================================================
def RePlotCommands_from_raster():
  # ================================================================================
  #  Load the data from the pickle file
  # ================================================================================
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']

  plot_from_raster(name, first_year, data, continent = "Africa")


# ================================================================================
#  Do only the mcmc for the validations
# ================================================================================
def mcmc_validations_only():
  #  Load the data from the pickle file
  file = open("Replot_dict.pck", "rb") # read mode
  Replot_dict = pickle.load(file)
  name = Replot_dict['name']
  data = Replot_dict['data']
  first_year = Replot_dict['first_year']
  thin = Replot_dict['thin']
  N1 = Replot_dict['N1']
  N2 = Replot_dict['N2']
  N3 = Replot_dict['N3']
  with_covariates = Replot_dict['with_covariates']
  first_year = 2019

  Validate = validate(data,name, N1, N2, N3, thin, first_year, with_covariates)


# ================================================================================
#  Do validation MCMCs one at a time
# ================================================================================
def mcmc_validations_only_one_at_a_time(i):
  #  Load the data from the pickle file
  file = open("validations_dict.pck", "rb") # read mode
  validations_dict = pickle.load(file)
  name = validations_dict['name']
  data = validations_dict['data']
  first_year = validations_dict['first_year']
  thin = validations_dict['thin']
  N1 = validations_dict['N1']
  N2 = validations_dict['N2']
  N3 = validations_dict['N3']
  with_covariates = validations_dict['with_covariates']
  choice = validations_dict['choice']
  first_year = 2019

  Validate = validate_single_group(data, name, N1, N2, N3, thin, first_year, choice, i, with_covariates,  continent = "Africa")



# ================================================================================
#  Redo the validations (bringing them together) from saved MCMC runs
# ================================================================================
def mcmc_validations_bring_together():
  #  Load the data from the pickle file
  file = open("validations_dict.pck", "rb") # read mode
  validations_dict = pickle.load(file)
  name = validations_dict['name']
  data = validations_dict['data']
  thin = validations_dict['thin']
  with_covariates = validations_dict['with_covariates']
  choice = validations_dict['choice']
  first_year = 2019

  redo_from_saved_validate(data, name, choice, thin, first_year, with_covariates, continent = "Africa")
