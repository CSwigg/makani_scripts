
import sys 
working_path = '/Users/cam/Desktop/astro_research/prospector_work'
sys.path.insert(0, working_path + '/sedpy')
sys.path.insert(0, working_path + '/prospector')
sys.path.insert(0, working_path + '/python_fsps_c3k')

import fsps
import sedpy
import prospect
import numpy as np

def build_sps(zcontinuous=1, sfh = 0, **extras):
    """
    :param zcontinuous: 
        A value of 1 insures that we use interpolation between SSPs to 
        have a continuous metallicity parameter (`logzsol`)
        See python-FSPS documentation for details
    """
    from prospect.sources import FastStepBasis

    logt_wmb_hot = np.log10(50000.0) # This basically turns off WMBasic
    use_wr_spectra = 0

    sps = FastStepBasis(zcontinuous=zcontinuous, sfh=sfh,
            use_wr_spectra = use_wr_spectra,
            logt_wmb_hot = logt_wmb_hot)

    return sps
