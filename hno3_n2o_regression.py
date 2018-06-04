
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linearinterp(arr, altrange):
    """
    :param arr: 2d array with dimensions [alt, time]
    :param altrange: array of altitudes corresponding to alt dimension of arr
    :return: copy of input arr that has missing values filled in
            by linear interpolation (over each altitude)
    """
    times = arr['Time']
    levels = arr['nLevels']
    arrinterp = np.zeros(np.shape(arr))
    for i in range(len(altrange)):
        y = pd.Series(arr[:, i])
        yn = y.interpolate(method='linear')
        arrinterp[:, i] = yn
    new_arr = xr.DataArray(arrinterp, coords=[times, levels], dims=['Time', 'nLevels'])
    return new_arr


if __name__ == "__main__":
    # load HNO3
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_hno3 = nox.groupby('Time.month') - monthlymeans
    anomalies_hno3['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_hno3 *= 1e9  # ppbv
    anomalies_hno3 = anomalies_hno3[:, 7:15]

    pressure_levels = datafile.Pressure.mean('Time')
    pressure_levels = pressure_levels[7:15]

    # Load N2O
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.N2O.where((datafile.Latitude > -5) & (datafile.Latitude < 5))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_n2o = nox.groupby('Time.month') - monthlymeans
    anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_n2o *= 1e9  # ppbv
    anomalies_n2o = anomalies_n2o[:, 7:15]

    # Interpolate missing values
    anomalies_n2o = linearinterp(anomalies_n2o, pressure_levels)
    anomalies_hno3 = linearinterp(anomalies_hno3, pressure_levels)

    reg_coeff = np.zeros(len(pressure_levels))
    sigma = np.zeros(len(pressure_levels))
    for i in range(len(pressure_levels)):
        lm = LinearRegression()
        lm.fit(anomalies_n2o[:, i].values.reshape((123, 1)), anomalies_hno3[:, i].values.reshape((123, 1)))
        y_pred = lm.predict(anomalies_n2o[:, i].values.reshape((123, 1)))
        reg_coeff[i] = lm.coef_

        mse = mean_squared_error(anomalies_hno3[:, i].values.reshape((123, 1)), y_pred)
        sigma[i] = np.sqrt(mse)
        print(r2_score(anomalies_hno3[:, i].values.reshape((123, 1)), y_pred))

    print(sigma)

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
    plt.title('MLS HNO3/N2O')
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/regression_HNO3_N2O.png", format='png', dpi=150)
    plt.show()

