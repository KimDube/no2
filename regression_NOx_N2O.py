
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NO2 import helper_functions, open_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    # Load daily NOx
    nox, pres_nox = open_data.load_osiris_nox_monthly(start_date='20050101', end_date='20141231',
                                                      min_lat=-10, max_lat=10, pressure=1)

    # Load N2O
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    no2 = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = no2.groupby('Time.month').mean('Time')
    anomalies_n2o = no2.groupby('Time.month') - monthlymeans
    anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_n2o *= 1e9  # ppbv
    anomalies_n2o = anomalies_n2o[:, 5:15]  # corresponds to range of pressures returned by alt_to_pres

    # Put NOx on MLS pressure levels to match HNO3
    nox_on_pres_levels = np.zeros((len(nox.time), 10))  # alt_to_pres returns 10 pressure levels
    mls_levels = []
    for i in range(len(nox.time)):
        nox_on_pres_levels[i, :], l = helper_functions.interpolate_to_mls_pressure(pres_nox[i, :], nox[i, :])
        mls_levels = l

    nox_dataset = xr.DataArray(nox_on_pres_levels, coords=[nox.time, mls_levels], dims=["Time", "nLevels"])
    monthlymeans = nox_dataset.groupby('Time.month').mean('Time')
    anomalies_nox = nox_dataset.groupby('Time.month') - monthlymeans

    # Interpolate missing values so regression works
    anomalies_n2o = helper_functions.linearinterp(anomalies_n2o, mls_levels)
    anomalies_nox = helper_functions.linearinterp(anomalies_nox, mls_levels)

    anomalies_n2o = anomalies_n2o[0:-3, :]

    reg_coeff = np.zeros(len(mls_levels))
    sigma = np.zeros(len(mls_levels))
    for i in range(len(mls_levels)):
        n2o_i = anomalies_n2o[:, i].reshape((120, 1))
        nox_i = anomalies_nox[:, i].reshape((120, 1))

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
    plt.semilogy(reg_coeff, mls_levels, '-o')
    # plt.errorbar(reg_coeff, mls_levels, xerr=sigma)
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

