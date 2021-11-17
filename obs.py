
import sys 
working_path = '/Users/cam/Desktop/astro_research/prospector_work'
sys.path.insert(0, working_path + '/sedpy')
sys.path.insert(0, working_path + '/prospector')
sys.path.insert(0, working_path + '/python_fsps_c3k')

import fsps

import sedpy
import prospect
from prospect.utils.obsutils import fix_obs
import numpy as np
import pandas as pd


def flux_to_maggies(wave, flux, flux_unc):
    maggies = flux*3.34e4*wave**2*1e-17/3631
    maggies_unc = flux_unc*3.34e4*wave**2*1e-17/3631
    return (maggies, maggies_unc)


def build_obs_spectra(object_data, object_redshift, object_spectrum = None, test_model = None, **extras):
    """Build a dictionary of observational data.  
    
    :returns obs:
    """
    
    data = object_data
    
    # The obs dictionary, empty for now
    obs = {}

    galex = ['galex_FUV', 'galex_NUV']
    sdss = ['sdss_{0}0'.format(b) for b in ['u','g','r','i','z']]
    wise = ['wise_W1', 'wise_W2', 'wise_W3', 'wise_W4']
    vista = ['vista_{0}'.format(b) for b in ['y', 'j', 'h', 'k']] 
    ukidss = ['ukidss_{0}'.format(b) for b in ['y', 'j', 'h', 'k']] 
    

    
    if (np.any(data['VISTA_VEGAMAG']) != 0.):
        filternames = galex + sdss + vista + wise
        obs['ukidss_or_vista'] = 'VISTA'

    elif (np.any(data['UKIDDS_VEGAMAG']) != 0.):
        filternames = galex + sdss + ukidss + wise
        obs['ukidss_or_vista'] = 'UKIDSS'

    else:
        filternames = galex + sdss + ukidss + wise
        obs['ukidss_or_vista'] = 'N/A'


    filternames = np.array(filternames)


    M_AB = np.array(data['AB_MAG'])
    flux_flam = np.array(data['FLUX_FLAM'])
    flux_flam_err = np.array(data['FLUX_FLAM_ERR'])

    
    indices_good = np.where((np.abs(M_AB) != 99.) & 
            (~np.isnan(M_AB)) & (M_AB > 15.00) & (flux_flam != 0.))
    

    flux_flam = flux_flam[indices_good[0]]
    flux_flam_err = flux_flam_err[indices_good[0]]
    
        
    wave = data['PHOT_WAVE'][indices_good[0]]
    obs['phot_wave'] = wave
    obs["filters"] = sedpy.observate.load_filters(filternames[indices_good[0]])

    maggies = (flux_flam*3.34e4*wave**2*1e-17)/(3631)
    maggies_unc = (flux_flam_err*3.34e4*wave**2*1e-17)/(3631)

    maggies, maggies_unc = flux_to_maggies(wave, flux_flam, flux_flam_err)
    print(maggies_unc)


    
    
    if object_spectrum is not None:
        spec = object_spectrum


        f = np.array(spec.data['flux'][0]).byteswap().newbyteorder()
        w = np.array(spec.data['wave'][0]).byteswap().newbyteorder()
        err = np.array(spec.data['err'][0]).byteswap().newbyteorder()
        mask = np.array(spec.data['mask'][0]).byteswap().newbyteorder()
        df = pd.DataFrame({'wspec':w, 'fspec':f, 'errspec':err, 'mask':mask})
        #dfg = df.loc[df['mask'] == 0.] 
        df = df.loc[(df['wspec'] >= 3500) & (df['wspec'] <= 4200)]

        fspec_maggies, fspec_err_maggies = flux_to_maggies(df.wspec.values, df.fspec.values, df.errspec.values)

    if 912*(1+object_redshift) > 1400:
        obs['phot_mask'] = np.array([('galex_FUV' not in f.name) for f in obs["filters"]])
        obs['galex_fuv_masked'] = True
    else:
        obs['galex_fuv_masked'] = False

    if test_model is not None:
        # if using a model as fake data
        maggies = test_model['flux_flam']
        fspec_maggies = test_model['fspec_maggies'] 
        obs["maggies"] = maggies
        obs["maggies_unc"] = maggies_unc

        obs["wavelength"] = df.wspec.values*(1+object_redshift)
        obs["spectrum"] = fspec_maggies
        obs['unc'] = fspec_err_maggies

    else:
        obs["maggies"] = maggies
        obs["maggies_unc"] = maggies_unc

        if object_spectrum is not None:
            obs["wavelength"] = df.wspec.values*(1+object_redshift)
            obs["spectrum"] = fspec_maggies*(1+object_redshift)
            obs['unc'] = fspec_err_maggies*(1+object_redshift)
            obs['mask'] = np.abs(df['mask'].values - 1)
        else:
            obs['wavelength'] = None
            obs['spectrum'] = None
            obs['unc'] = None
    obs = fix_obs(obs)

    return obs 

