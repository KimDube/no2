###
# Correlate NOy=NOx+HNO3 and O3
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats.stats import pearsonr
import numpy as np
import pandas as pd


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


def linearinterp(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: copy of input arr that has missing values filled in
            by linear interpolation (over each altitude)
    """
    arrinterp = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[i, :])
        yn = y.interpolate(method='linear')
        arrinterp[i, :] = yn
    return arrinterp


if __name__ == "__main__":
    # Load NOx
    dataf = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    dataf = dataf.swap_dims({'profile_id': 'time'}, inplace=True)
    dataf = dataf.sel(time=slice('20050101', '20141231'))
    nox = dataf.derived_0630_NOx_concentration.where((dataf.latitude > -10) & (dataf.latitude < 10))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    nox *= 6.022140857e17
    # To convert number density to vmr
    pres = dataf.pressure.where((dataf.latitude > -10) & (dataf.latitude < 10))
    temp = dataf.temperature.where((dataf.latitude > -10) & (dataf.latitude < 10))
    nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
    nox = nox.resample('MS', dim='time', how='mean')
    monthlymeans = nox.groupby('time.month').mean('time')
    anomalies_nox = (nox.groupby('time.month') - monthlymeans)
    anomalies_nox = anomalies_nox
    times_nox = anomalies_nox.time.values
    alts_nox = anomalies_nox.altitude.values

    # Load HNO3
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_hno3 = nox.groupby('Time.month') - monthlymeans
    anomalies_hno3['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_hno3 *= 1e9  # ppbv

    # Calculate NOy = HNO3 + NOx
    nox_pres = pres.mean(dim='time')
    mls_levels = datafile.Pressure.mean('Time')
    heights_index = []
    for i in mls_levels[7:15]:
        n = find_nearest(nox_pres, i)
        heights_index.append(int(n.values))

    anomalies_noy = np.zeros((np.shape(anomalies_nox)[0], len(mls_levels[7:15])))
    for i in range(np.shape(anomalies_nox)[0]):
        anomalies_noy[i, :] = anomalies_hno3.values[i, 7:15] + anomalies_nox.values[i, heights_index]

    # Load N2O
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    n2o = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = n2o.groupby('Time.month').mean('Time')
    anomalies_n2o = n2o.groupby('Time.month') - monthlymeans
    anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_n2o *= 1e9  # ppbv
    anomalies_n2o = anomalies_n2o.values

    # Interpolate missing values, otherwise correlation won't work
    anomalies_noy = linearinterp(anomalies_noy, mls_levels.values[7:15])
    anomalies_n2o = linearinterp(anomalies_n2o, mls_levels.values[7:15])

    # Calculate corr coeff for each pressure level
    cc = np.zeros(len(heights_index))
    for i in range(len(heights_index)):
        anomalies_n2o[:, i] = (anomalies_n2o[:, i] - np.nanmean(anomalies_n2o[:, i])) / np.nanstd(anomalies_n2o[:, i])
        anomalies_noy[:, i] = (anomalies_noy[:, i] - np.nanmean(anomalies_noy[:, i])) / np.nanstd(anomalies_noy[:, i])
        cc[i], p = pearsonr(anomalies_noy[:, i], anomalies_n2o[:-3, i])

    # plot
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta', 'tangerine', 'medium green']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot([0, 0], [4, 73], sns.xkcd_rgb["charcoal grey"])
    plt.semilogy(cc, mls_levels.values[7:15], '-o')
    plt.ylabel('Pressure [km]')
    plt.xlabel('Correlation Coefficient')
    plt.xlim([-1, 1])
    plt.ylim([4, 73])
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(plt.FixedLocator([4.6, 6.8, 10.0, 14.7, 21.5, 31.6, 46.4, 68.1]))
    plt.gca().invert_yaxis()
    ax.grid(True, which="major", ls="-", axis="y")
    plt.title('MLS N2O vs. NOy=OSIRIS NOx + MLS HNO3')
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/corr_N2O_NOx.png", format='png', dpi=150)
    plt.show()