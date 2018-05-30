###
# Correlate NOx and O3
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from scipy.stats.stats import pearsonr
import numpy as np


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
    anomalies_nox = anomalies_nox.values

    # Load O3
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/CCI/OSIRIS_v5_10/*.nc')
    datafile = datafile.sel(time=slice('20050101', '20141231'))
    o3 = datafile.ozone_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    o3 *= 6.022140857e17
    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
    temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))
    o3 = (o3 * temp * 1.3806503e-19 / pres) * 1e6  # ppmv
    o3 = o3.resample('MS', dim='time', how='mean')
    monthlymeans = o3.groupby('time.month').mean('time')
    anomalies_o3 = o3.groupby('time.month') - monthlymeans
    times_o3 = anomalies_o3.time.values
    alts_o3 = anomalies_o3.altitude.values
    anomalies_o3 = anomalies_o3.values

    cc = np.zeros(len(alts_nox))
    for i in range(len(alts_nox)):
        anomalies_o3[:, i] = (anomalies_o3[:, i] - np.nanmean(anomalies_o3[:, i])) / np.nanstd(anomalies_o3[:, i])
        anomalies_nox[:, i] = (anomalies_nox[:, i] - np.nanmean(anomalies_nox[:, i])) / np.nanstd(anomalies_nox[:, i])
        cc[i], p = pearsonr(anomalies_nox[:, i], anomalies_o3[:, i])

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta', 'tangerine', 'medium green']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.plot(cc, alts_nox, '-o', label='NO$\mathregular{_x}$ [ppbv]')
    plt.ylabel('Altitude [km]')
    plt.xlabel('Correlation Coefficient')
    plt.xlim([-1, 1])
    # plt.title('Monthly Mean OSIRIS Anomaly @ 6:30, 32 km (-10 to 10 deg. lat.)')
    # plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/corr_O3_NOx.png", format='png', dpi=150)
    plt.show()