from grb_common.extinction import extinction_curve
from grb_common.constants import C
import numpy as np

def opt_extinction(mag_data,mag_err,frequency,Rv,Ebv,zeropointflux):
    
    # C is speed of light in cm/s (from grb_common)
    # Convert frequency (Hz) to wavelength (Angstrom)
    wave = np.array([C/frequency*1e8])
    Av = Rv * Ebv
    
    # Calculate extinction using grb_common
    ext_mag = extinction_curve(wave, av=Av, rv=Rv, law='Fitzpatrick99')
    
    mag_data_deredden = mag_data - ext_mag
    flux_data_deredden = 10**(0.4*(zeropointflux-mag_data_deredden))
    flux_data_err = 0.4*np.log(10.0)*flux_data_deredden*mag_err
    
    return flux_data_deredden, flux_data_err
