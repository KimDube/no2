
import numpy as np
import pandas as pd
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import interpolate


def relative_anomaly(monthly_dataset):
    """
    Calculate relative anomaly.
    :param monthly_dataset: a dataset with monthly spacing in time
    :return: relative anomaly from monthly mean for each value in dataset
    """
    monthlymeans = monthly_dataset.groupby('time.month').mean('time')
    monthly_anom = monthly_dataset.groupby('time.month') - monthlymeans
    monthly_anom = monthly_anom.groupby('time.month') / monthlymeans
    return monthly_anom


def interpolate_to_mls_pressure(osiris_pressure, osiris_vmr):
    """
    Interpolate OSIRIS data from altitude to MLS pressure levels 5 to 15.
    :param osiris_pressure: 1d OSIRIS pressures
    :param osiris_vmr: Corresponding 1D OSIRIS VMR (NOx, O3, etc.)
    :return: OSIRIS VMR on MLS levels, array of corresponding pressures
    """
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


def linearinterp2d(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: copy of input arr that has missing values filled in
            by linear interpolation (over each altitude)
    """
    arrinterp = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[:, i])
        yn = y.interpolate(method='linear')
        arrinterp[:, i] = yn
    return arrinterp


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


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

