"""
Interpolate OSIRIS data from altitude to MLS pressure level
"""


import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate


def interpolate_to_mls_pressure(osiris_pressure, osiris_vmr):
    mls_levels = np.load('/home/kimberlee/Masters/NO2/mls_pressure_levels.npy')
    # Interested in 100 to 5 hPa
    mls_levels = mls_levels[5:15]

    # pres_to_alt is equation describing relationship between OSIRIS altitude and pressure.
    # Feed it an MLS pressure level to get the corresponding altitude
    pres_to_alt = interpolate.interp1d(osiris_pressure, osiris_pressure.altitude)
    mls_levels_alt = pres_to_alt(mls_levels)

    # interp_nox is equation describing relationship between NOx density and altitude. Feed it
    # an altitude to get the density.
    interp_nox = interpolate.interp1d(osiris_vmr.altitude, osiris_vmr)
    nox_on_mls_levels = interp_nox(mls_levels_alt)
    return nox_on_mls_levels, mls_levels


if __name__ == "__main__":
    # load NOx
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20050101', '20051231'))
    nox = datafile.derived_daily_mean_NOx_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    nox *= 6.022140857e17
    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
    temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))
    nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv

    nox = nox.resample('MS', dim='time', how='mean')
    pres = pres.resample('MS', dim='time', how='mean')

    # for each monthly mean profile
    for i in range(len(nox.time)):
        pressure_i = pres[i, :]
        nox_i = nox[i, :]
        n, l = interpolate_to_mls_pressure(pressure_i, nox_i)

        sns.set(context="talk", style="white", rc={'font.family': [u'serif']})

        plt.loglog(nox_i, pressure_i, '-*')
        plt.loglog(n, l, 'o')

        # for i in mls_levels:
        #    plt.semilogy([10, 40], [i, i], '-k', linewidth=0.5)

        plt.ylabel("Pressure [hPa]")
        #plt.ylim([2, 500])
        plt.gca().invert_yaxis()
        plt.show()

