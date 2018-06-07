###
# Correlate NOy=NOx+HNO3 and O3
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats.stats import pearsonr
import numpy as np
from NO2 import helper_functions, open_data


if __name__ == "__main__":
    # Load daily NOx
    nox, pres_nox = open_data.load_osiris_nox_monthly(start_date='20050101', end_date='20141231',
                                                      min_lat=-10, max_lat=10, pressure=1)

    # Load monthly mean HNO3
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    hno3 = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    hno3['nLevels'] = datafile.Pressure.mean('Time')  # change level numbers to appropriate pressure
    hno3 *= 1e9  # ppbv

    # Put NOx on MLS pressure levels to match HNO3
    nox_on_pres_levels = np.zeros((len(nox.time), 10))  # alt_to_pres returns 10 pressure levels
    mls_levels = []
    for i in range(len(nox.time)):
        nox_on_pres_levels[i, :], l = helper_functions.interpolate_to_mls_pressure(pres_nox[i, :], nox[i, :])
        mls_levels = l

    # Calculate NOy = HNO3 + NOx
    nox_dataset = xr.DataArray(nox_on_pres_levels, coords=[nox.time, mls_levels], dims=["Time", "nLevels"])
    noy = np.zeros((len(nox_dataset.Time), 10))
    for i in range(len(nox_dataset.Time)):
        noy[i, :] = hno3.values[i, 5:15] + nox_dataset.values[i, :]

    noy_dataset = xr.DataArray(noy, coords=[nox_dataset.Time, mls_levels], dims=["Time", "nLevels"])
    monthlymeans = noy_dataset.groupby('Time.month').mean('Time')
    anomalies_noy = noy_dataset.groupby('Time.month') - monthlymeans

    # Load O3
    o3, pres_o3 = open_data.load_osiris_ozone_monthly(start_date='20050101', end_date='20141231',
                                                      min_lat=-10, max_lat=10, pressure=1)

    # Put O3 on MLS pressure levels to match HNO3
    o3_on_pres_levels = np.zeros((len(o3.time), 10))  # alt_to_pres returns 10 pressure levels
    for i in range(len(o3.time)):
        o3_on_pres_levels[i, :], l = helper_functions.interpolate_to_mls_pressure(pres_o3[i, :], o3[i, :])

    # Calculate NOy = HNO3 + NOx
    o3_dataset = xr.DataArray(o3_on_pres_levels, coords=[o3.time, mls_levels], dims=["Time", "nLevels"])
    monthlymeans = o3_dataset.groupby('Time.month').mean('Time')
    anomalies_o3 = o3_dataset.groupby('Time.month') - monthlymeans

    # Interpolate missing values, otherwise correlation won't work
    anomalies_noy = helper_functions.linearinterp(anomalies_noy, mls_levels)
    anomalies_o3 = helper_functions.linearinterp(anomalies_o3, mls_levels)

    # Calculate corr coeff for each pressure level
    cc = np.zeros(len(mls_levels))
    for i in range(len(mls_levels)):
        # Standardize so result is between -1 and 1
        anomalies_o3[:, i] /= np.nanstd(anomalies_o3[:, i])
        anomalies_noy[:, i] /= np.nanstd(anomalies_noy[:, i])
        cc[i], p = pearsonr(anomalies_noy[:, i], anomalies_o3[:, i])

    # Plot
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta', 'tangerine', 'medium green']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot([0, 0], [4, 73], sns.xkcd_rgb["charcoal grey"])
    plt.semilogy(cc, mls_levels, '-o')
    plt.ylabel('Pressure [hPa]')
    plt.xlabel('Correlation Coefficient')
    plt.xlim([-1, 1])
    plt.ylim([4, 73])
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(plt.FixedLocator(mls_levels))
    plt.gca().invert_yaxis()
    ax.grid(True, which="major", ls="-", axis="y")
    plt.title('OSIRIS O3 vs. NOy=OSIRIS NOx + MLS HNO3')
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/corr_O3_NOy.png", format='png', dpi=150)
    plt.show()
