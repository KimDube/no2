
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


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
    #times = arr['Time']
    #levels = arr['nLevels']
    arrinterp = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[:, i])
        yn = y.interpolate(method='linear')
        arrinterp[:, i] = yn
    #new_arr = xr.DataArray(arrinterp, coords=[times, levels], dims=['Time', 'nLevels'])
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
    times_nox = anomalies_nox.time.values
    alts_nox = anomalies_nox.altitude.values

    # Load N2O
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.N2O.where((datafile.Latitude > -5) & (datafile.Latitude < 5))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_n2o = nox.groupby('Time.month') - monthlymeans
    anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_n2o *= 1e9  # ppbv
    anomalies_n2o = anomalies_n2o[:, 7:15]

    pressure_levels = datafile.Pressure.mean('Time')
    pressure_levels = pressure_levels[7:15]

    nox_pres = pres.mean(dim='time')
    heights_index = []
    for i in pressure_levels:
        n = find_nearest(nox_pres, i)
        heights_index.append(int(n.values))

    anomalies_nox_pres = anomalies_nox[:, heights_index]

    # anomalies_nox_pres = anomalies_nox_pres.transpose()
    # Interpolate missing values
    anomalies_n2o = linearinterp(anomalies_n2o, pressure_levels)
    anomalies_nox_pres = linearinterp(anomalies_nox_pres, pressure_levels)

    anomalies_n2o = anomalies_n2o[0:-3, :]

    reg_coeff = np.zeros(len(pressure_levels))
    sigma = np.zeros(len(pressure_levels))
    for i in range(len(pressure_levels)):
        n2o_i = anomalies_n2o[:, i].reshape((120, 1))
        nox_i = anomalies_nox_pres[:, i].reshape((120, 1))

        lm = LinearRegression()
        lm.fit(n2o_i, nox_i)
        y_pred = lm.predict(n2o_i)
        # print(lm.coef_)
        reg_coeff[i] = lm.coef_[0]

        mse = mean_squared_error(nox_i, y_pred)
        sigma[i] = np.sqrt(mse)
        # print(r2_score(anomalies_nox_pres[:, i], y_pred))

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta', 'tangerine', 'medium green']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot([0, 0], [4, 73], sns.xkcd_rgb["charcoal grey"])
    plt.semilogy(reg_coeff, pressure_levels, '-o')
    # plt.errorbar(reg_coeff, pressure_levels, xerr=sigma)
    plt.ylabel('Pressure [hPa]')
    plt.xlabel('Regression Coefficient (HNO3/N2O)')
    plt.xlim([-0.06, 0.02])
    plt.ylim([4, 73])
    from matplotlib.ticker import ScalarFormatter

    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(plt.FixedLocator([4.6, 6.8, 10.0, 14.7, 21.5, 31.6, 46.4, 68.1]))
    plt.gca().invert_yaxis()
    ax.grid(True, which="major", ls="-", axis="y")
    plt.title('MLS NOx/N2O')
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/regression_NOx_N2O.png", format='png', dpi=150)
    plt.show()

