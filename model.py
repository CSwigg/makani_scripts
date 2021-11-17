'''
Model that adopts a non-parametric continuity SFH, two component dust attenuation,
and dust emission. 
'''

import sys 
working_path = '/Users/cam/Desktop/astro_research/prospector_work'
sys.path.insert(0, working_path + '/sedpy')
sys.path.insert(0, working_path + '/prospector')
sys.path.insert(0, working_path + '/python_fsps_c3k')
#sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/prospector')
#sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/python_fsps_c3k')
#sys.path.insert(0, '/Users/cam/Desktop/astro_research/prospector_work/p_scripts')

import fsps

import sedpy
import prospect
import numpy as np
import pandas as pd

df = pd.read_csv('/Users/cam/Desktop/astro_research/prospector_work/add_sigma_c3k.csv')

def build_model_latest(g_name, obs, object_redshift=None, ldist=10.0, fixed_metallicity=None, add_duste=False,
                **extras):

    from prospect.models.sedmodel import SedModel, SpecModel, PolySpecModel
    from prospect.models.templates import TemplateLibrary
    from prospect.models import priors, transforms

    model_params = TemplateLibrary["continuity_sfh"]

    # spectral smoothing
    n_params = 0
    obs['wavelength'] = None
    if obs['wavelength'] is not None:
        add_sigma = df[df['short_name'] == g_name]['add_sigma'].values
        model_params.update(TemplateLibrary['spectral_smoothing'])
        model_params['sigma_smooth']['isfree'] = True
        model_params['sigma_smooth']['init'] = add_sigma
        model_params['sigma_smooth']['prior'] = priors.Uniform(mini=150, maxi=250)
        n_params += 1


    #obs['ukidss_or_vista'] = "None"
    #if obs['ukidss_or_vista'] == 'UKIDSS':
    #    init = [[f for f in obs['filternames'] if 'ukdss' in f]][0]

    #if obs['ukidss_or_vista'] == 'VISTA':
    #    init = [[f for f in obs['filternames'] if 'vista' in f]][0]

    #if (obs['ukidss_or_vista'] == 'UKDISS') or (obs['ukidss_or_vista'] == 'VISTA'):

    #    # NoiseModel Parameters
    #    model_params['phot_jitter'] = dict(N=1, isfree=False, init=1.0,
    #                                       units="scale the photometric noise by this factor",
    #                                       prior=None)
    #    model_params['nir_offset'] = dict(N=1, isfree=True, init=1.0,
    #                                      units="fractional offset of the NIR bands",
    #                                      prior=priors.TopHat(mini=0.5, maxi=2))
    #    model_params['nir_bandnames'] = dict(N=3, isfree=False,
    #                                         init=init,
    #                                         units="name of the correlated NIR bands",
    #                                         prior=None)

    #    n_params += 22
    #else:
    #    n_params += 21



    nbins_sfh = 6
    model_params['agebins']['N'] = nbins_sfh
    model_params['mass']['N'] = nbins_sfh
    model_params['logsfr_ratios']['N'] = nbins_sfh-1
    model_params['logsfr_ratios']['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    
    #model_params['logsfr_ratios']['prior'] = priors.ClippedNormal(mean=np.full(nbins_sfh-1,0.0),
    #                                                              sigma=np.full(nbins_sfh-1,0.3),
    #                                                              mini=np.full(nbins_sfh-1,-0.3),
    #                                                              maxi=np.full(nbins_sfh-1,.3)
    #                                                              )

    model_params['logsfr_ratios']['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                  scale=np.full(nbins_sfh-1,0.6),
                                                                  df=np.full(nbins_sfh-1,2))
    
    
    model_params["dust1"] = {"N": 1, "isfree": False, 'depends_on': transforms.dustratio_to_dust1,
                        "init": 1., "units": "optical depth towards young stars"}

    model_params["dust_ratio"] = {"N": 1, "isfree": True, 
                             "init": 1.0, "units": "ratio of birth-cloud to diffuse dust",
                             "prior": priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)}

    model_params["dust_index"] = {"N": 1, "isfree": True,
                             "init": 0.0, "units": "power-law multiplication of Calzetti",
                             "prior": priors.TopHat(mini=-2.0, maxi=1.)}
    
    log_agebins = np.array([[0,6],
                            [6, 6.5],
                            [6.5, 7],
                            [7,8],
                            [8,9],
                            [9,10]])

    #log_agebins = np.array([[0, 6.],
    #                        [6., 6.5],
    #                        [6.5, 6.75], 
    #                        [6.75, 7.],
    #                        [7., 7.25],
    #                        [7.25, 7.5],
    #                        [7.5, 7.75],
    #                        [7.75, 8.],
    #                        [8., 8.25],
    #                        [8.25, 8.5],
    #                        [8.5, 8.75],
    #                        [8.75, 9.],
    #                        [9., 9.5],
    #                        [9.5, 10.]
    #                        ])


    model_params['agebins']['init'] = log_agebins
    
    model_params["logmass"]["init"] = 11.
    model_params["dust2"]["init"] = 0.5
    model_params["logzsol"]["init"] = 0.0

    
    model_params["logmass"]["prior"] = priors.TopHat(mini=9.5, maxi=12.)
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["logzsol"]["prior"] = priors.TopHat(mini=-0.5, maxi=0.5)
    
    
    if fixed_metallicity is not None:
        model_params["logzsol"]["isfree"] = False
        model_params["logzsol"]['init'] = fixed_metallicity 

    if object_redshift is not None:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = object_redshift

    if add_duste:        
        model_params.update(TemplateLibrary["dust_emission"])
        model_params['duste_umin']['isfree'] = True
        model_params['duste_qpah']['isfree'] = True
        model_params['duste_gamma']['isfree'] = True
        
        model_params['duste_umin']['prior'] = priors.LogUniform(mini=0.1, maxi=15.)
        model_params['duste_qpah']['prior'] = priors.LogUniform(mini=0.1, maxi=5.)
        model_params['duste_gamma']['prior'] = priors.LogUniform(mini=0.001, maxi=0.15)
        
        model_params['duste_umin']['init'] = 7.5
        model_params['duste_qpah']['init'] = 3.5
        model_params['duste_gamma']['init'] = 0.075

        model_params["duste_qpah"]["disp_floor"] = 3.0
        model_params["duste_umin"]["disp_floor"] = 4.5
        model_params["duste_gamma"]["disp_floor"] = 0.15
        model_params["duste_qpah"]["init_disp"] = 3.0
        model_params["duste_umin"]["init_disp"] = 5.0
        model_params["duste_gamma"]["init_disp"] = 0.2
        
    model_params['dust2']['disp_floor'] = 1e-2 
    model_params['dust1']['disp_floor'] = 1e-2 
    model_params['logzsol']['disp_floor'] = 1e-3
    model_params['dust_index']['disp_floor'] = 1e-3


    model = SpecModel(model_params)

    return model, n_params

