"""
Compatibility layer for migrating to grb-common.

This module provides backward-compatible functions that use grb-common
when available, falling back to the original implementations otherwise.

Usage:
    # Replace: from extinc import opt_extinction
    from compat import opt_extinction

    # Replace: from grb_utils import get_cosmology
    from compat import luminosity_distance, cosmology
"""

import numpy as np

# Try to import from grb-common, fall back to local implementations
try:
    from grb_common.extinction import deredden_flux, fitzpatrick99
    from grb_common.cosmology import luminosity_distance, get_cosmology
    from grb_common.constants import C_LIGHT, M_ELECTRON, SIGMA_T, MPC
    from grb_common.fitting import (
        UniformPrior,
        LogUniformPrior,
        GaussianPrior,
        CompositePrior,
        chi_squared,
        gaussian_likelihood,
    )

    HAS_GRB_COMMON = True
    print("Using grb-common for shared utilities")

except ImportError:
    HAS_GRB_COMMON = False
    print("grb-common not installed, using local implementations")

    # Physical constants (CGS)
    C_LIGHT = 2.99792458e10  # cm/s
    M_ELECTRON = 9.1093837e-28  # g
    SIGMA_T = 6.6524587e-25  # cm^2
    MPC = 3.085678e24  # cm

    def luminosity_distance(z, H0=70, Om0=0.3):
        """Approximate luminosity distance for flat LCDM."""
        from astropy.cosmology import FlatLambdaCDM
        from astropy import units as u
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        return cosmo.luminosity_distance(z).to(u.cm).value

    def fitzpatrick99(wavelength, Av, Rv=3.1):
        """Fitzpatrick99 extinction."""
        from extinction import fitzpatrick99 as f99
        return f99(np.atleast_1d(wavelength), Av, Rv)


def opt_extinction(mag_data, mag_err, frequency, Rv, Ebv, zeropointflux):
    """
    Apply optical extinction correction.

    Parameters
    ----------
    mag_data : float or array
        Observed magnitudes.
    mag_err : float or array
        Magnitude errors.
    frequency : float
        Observation frequency in Hz.
    Rv : float
        R_V extinction parameter.
    Ebv : float
        E(B-V) color excess.
    zeropointflux : float
        Zero-point flux for magnitude system.

    Returns
    -------
    flux_deredden : array
        Dereddened flux.
    flux_err : array
        Flux error.
    """
    # Convert frequency to wavelength in Angstrom
    wave = np.array([C_LIGHT / frequency * 1e8])
    Av = Rv * Ebv

    if HAS_GRB_COMMON:
        A_lambda = fitzpatrick99(wave, Av=Av, Rv=Rv)
    else:
        from extinction import fitzpatrick99 as f99
        A_lambda = f99(wave, Av, Rv)

    mag_data_deredden = mag_data - A_lambda
    flux_data_deredden = 10**(0.4 * (zeropointflux - mag_data_deredden))
    flux_data_err = 0.4 * np.log(10.0) * flux_data_deredden * mag_err

    return flux_data_deredden, flux_data_err


def make_priors(param_bounds):
    """
    Create prior distribution from parameter bounds.

    Parameters
    ----------
    param_bounds : dict
        Parameter bounds as {name: (low, high, log_scale)}.

    Returns
    -------
    priors : CompositePrior or dict
        Prior distribution object or dict of bounds.
    """
    if HAS_GRB_COMMON:
        prior_dict = {}
        for name, (low, high, log_scale) in param_bounds.items():
            if log_scale:
                prior_dict[name] = LogUniformPrior(low, high)
            else:
                prior_dict[name] = UniformPrior(low, high)
        return CompositePrior(prior_dict)
    else:
        return param_bounds


__all__ = [
    'HAS_GRB_COMMON',
    'C_LIGHT',
    'M_ELECTRON',
    'SIGMA_T',
    'MPC',
    'luminosity_distance',
    'fitzpatrick99',
    'opt_extinction',
    'make_priors',
]
