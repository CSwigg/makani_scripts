import time, sys
import argparse

import os
# os.environ["OMP_NUM_THREADS"] = "1"

# import multiprocessing as mp
# mp.set_start_method('fork')
# from multiprocessing import Pool

import h5py
import numpy as np
import scipy
from matplotlib.pyplot import *
from astropy.io import fits
from astropy.cosmology import LambdaCDM
np.seterr(divide='ignore')
import warnings
warnings.filterwarnings('ignore')

style.use('default')

from matplotlib.font_manager import FontProperties
from matplotlib import gridspec


import sys
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python_fsps_c3k')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/sedpy')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_models') # My directory of models
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_sps') # My directory of sps
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/cam_obs') # My directory of obs
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/stats') # My directory of stat functions
sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/scripts/plotting_scripts') # My directory of plotting functions

import fsps
sps = fsps.StellarPopulation(zcontinuous=1)
print(sps.libraries) # TEST 

import sedpy
import prospect
from prospect.utils.obsutils import fix_obs
from helper_functions import * 

from non_parametric_model_1 import *
from non_parametric_model_spec_cal import *
from non_parametric_model_add_bin import *
from non_parametric_model_nir_fix import *
from non_parametric_model_latest import *
from fast_step_basis_sps import *
from obs_1_spectra import *
#from obs_1 import *
from calc_likelihood import *
from plot import *

import emcee
import dynesty

print(fsps)

def generate_model(theta, obs, sps, model):

    mspec, mphot, mextra = model.mean_model(theta, obs, sps=sps)

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    wphot = obs["phot_wave"]
    if obs["wavelength"] is None:
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]
                
    return wspec, mspec, mphot, wphot, mextra


def read_in_model(filepath):
    
    result, obs, _ = reader.results_from(filepath, dangerous=False)
    run_params = result['run_params']
    
    sps = build_sps(**run_params)
    model = build_model(**run_params)

    imax = np.argmax(result['lnprobability'])
    i, j = np.unravel_index(imax, result['lnprobability'].shape)
    theta_max = result['chain'][i, j, :].copy()
    
    wspec, mspec, mphot, wphot, mextra = generate_model(theta_max, obs, sps, model)
   
    test_model = {}
    test_model['flux_flam'] = mphot
    test_model['fspec_maggies'] = mspec

    return test_model




hizea_file = fits.open('/Users/cam/Desktop/astro_research/prospector_work/hizea_photo_galex_wise_v1.3.fit')[1]
cosmo = LambdaCDM(67.4, .315, .685)

run_directory = '/Users/cam/Desktop/astro_research/prospector_work/results/test_c3k_mpi_test/'

galaxies = hizea_file.data
#galaxies = [galaxies[1]]

start_time = time.time()

for i in range(len(galaxies)):
   

    galaxy = galaxies[i]
    galaxy_name = galaxy['short_name']

    #if galaxy_name not in ['J0901+0314','J0905+5759','J0826+4305']:
    #    continue


    
    spec = fits.open('/Users/cam/Desktop/astro_research/prospector_work/spectra/{}_mask.fit'.format(galaxy_name))[1]
    
    galaxy_z = galaxy['Z']
    galaxy_z_age = cosmo.age(galaxy_z) # age at given z
    
    galaxy_output_directory = run_directory + galaxy_name
    
    try:
        os.mkdir(run_directory + galaxy_name)
    except:
        continue 
        
    print('Running on {}'.format(galaxy_name))

    run_params = {}
    object_data = galaxy
    
    run_params['object_redshift'] = galaxy_z
    run_params['object_redshift_age'] = galaxy_z_age
    run_params["fixed_metallicity"] = None
    run_params["zcontinuous"] = 1
    run_params["sfh"] = 3
    #run_params["verbose"] = verbose
    run_params["add_duste"] = True
    run_params['g_name'] = galaxy_name

    # setting temperature
    run_params["logt_wmb_hot"] = np.log10(50000.0)
    run_params["use_wr_spectra"] = 0
    
    
    
    # testing on the original best-fit model (stability test)
    #test_model = read_in_model('/Users/cam/Desktop/astro_research/prospector_work/results/test_spec_model_2/{}/{}.h5'.format(galaxy_name, galaxy_name))
    

 
    #n_params = 22
    #n_params = 18
    #n_params = 19
    #n_params = 20

    obs = build_obs_spectra(object_data = object_data, object_redshift = galaxy_z, object_spectrum = spec, test_model = None)
    #obs = build_obs(object_data = object_data, object_redshift = galaxy_z)
    sps = build_sps(**run_params)
    model, n_params = build_model_latest(**run_params, obs = obs)
    print(n_params)
    #model = build_model(**run_params)
    
    # Prospector imports; MUST BE IMPORTED AFTER DEFINING SPS, otherwise segfault occurs
    from prospect.fitting import lnprobfn, fit_model
    from prospect.io import write_results as writer
    import prospect.io.read_results as reader
    from prospect.likelihood import chi_spec, chi_phot

    a = 1.0 + model.params.get('zred', 0.0) # cosmological redshifting
    wphot = obs["phot_wave"]
    if obs["wavelength"] is None:
        wspec = sps.wavelengths
        wspec *= a #redshift them
    else:
        wspec = obs["wavelength"]



    # ------- MCMC sampling -------
    run_params["optimize"] = False
    run_params["emcee"] = True
    run_params["dynesty"] = False

    run_params["nwalkers"] = 200
    run_params["niter"] = 10000
    run_params["nburn"] = [800, 800, 1000]

    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('done with dynesty in {0}s'.format(output["sampling"][1]))

    hfile = galaxy_output_directory + '/' + "{}.h5".format(galaxy_name)
    print(hfile)
    writer.write_hdf5(hfile, run_params, model, obs,
                      output["sampling"][0], output["optimization"][0],
                      tsample=output["sampling"][1],
                      toptimize=output["optimization"][1])
    
    
    print('Fitting of {} finished'.format(galaxy_name) + '\n')

print("--- %s hours ---" % ((time.time() - start_time)/60/60))


