

#------------ change these path variables ----------------
# this is where the scripts are pointed to
working_path = '/Users/cam/Desktop/astro_research/prospector_work'

# this is where the results are stored
run_directory = '/Users/cam/Desktop/astro_research/prospector_work/results/makani_results/' 
#---------------------------------------------------------



import time, sys
import argparse
import os
import h5py
import numpy as np
import scipy
from matplotlib.pyplot import *
from astropy.io import fits
from astropy.cosmology import LambdaCDM
import warnings

import sys 
sys.path.insert(0, working_path + '/sedpy')
sys.path.insert(0, working_path + '/prospector')
sys.path.insert(0, working_path + '/python_fsps_c3k')

import fsps
sps = fsps.StellarPopulation(zcontinuous=1)

import sedpy
import prospect
from prospect.utils.obsutils import fix_obs
from obs import build_obs
from sps import build_sps
from model import build_model

import emcee
import dynesty


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




hizea_file = fits.open('/Users/cam/Desktop/astro_research/prospector_work/hizea_fixed.fit')[1] # TODO: change this
cosmo = LambdaCDM(67.4, .315, .685)


galaxies = hizea_file.data
#galaxies = [galaxies[1]]

start_time = time.time()

for i in range(len(galaxies)):
   

    galaxy = galaxies[i]
    galaxy_name = galaxy['short_name']

    # these two lines make sure only Makani is run on
    if galaxy_name not in ['J2118+0017']:
        continue
    
    #spec = fits.open('/Users/cam/Desktop/astro_research/prospector_work/spectra/{}_mask.fit'.format(galaxy_name))[1]
    
    galaxy_z = galaxy['Z']
    galaxy_z_age = cosmo.age(galaxy_z) # age at given z
    
    galaxy_output_directory = run_directory + galaxy_name
    
    try:
        os.mkdir(run_directory + galaxy_name)
    except:
        print('Results directory not found or already created')
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
    

    obs = build_obs(object_data = object_data, object_redshift = galaxy_z, object_spectrum = None, test_model = None)
    sps = build_sps(**run_params)
    model, n_params = build_model(**run_params, obs = obs)
    
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


