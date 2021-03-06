###
# NOx and O3 at 10 hPa
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np
from NO2 import helper_functions, open_data


if __name__ == "__main__":
    # Load N2O
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_n2o = nox.groupby('Time.month') - monthlymeans
    anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_n2o *= 1e9  # ppbv
    anomalies_n2o = anomalies_n2o.sel(nLevels=10)  # 10 hPa

    # Load HNO3
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    nox = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies_hno3 = nox.groupby('Time.month') - monthlymeans
    anomalies_hno3['nLevels'] = datafile.Pressure.mean('Time')
    anomalies_hno3 *= 1e9  # ppbv
    anomalies_hno3 = anomalies_hno3.sel(nLevels=10)  # 10 hPa

    # Load NOx
    nox, pres_nox = open_data.load_osiris_nox_monthly(start_date='20050101', end_date='20141231',
                                                      min_lat=-10, max_lat=10, pressure=1)

    # get values at 10 hPa
    nox_10_hpa = np.zeros(len(nox.time))
    for i in range(len(nox.time)):
        pressure_i = pres_nox[i, :]
        nox_i = nox[i, :]
        n, l = helper_functions.interpolate_to_mls_pressure(pressure_i, nox_i)
        # only keep vmr at 10 hPa
        nox_10_hpa[i] = n[np.where(l == 10)]

    nox_10_hpa_dataset = xr.DataArray(nox_10_hpa, coords=[nox.time], dims=["time"])
    monthlymeans = nox_10_hpa_dataset.groupby('time.month').mean('time')
    anomalies_nox = nox_10_hpa_dataset.groupby('time.month') - monthlymeans

    # Load O3
    o3, pres_o3 = open_data.load_osiris_ozone_monthly(start_date='20050101', end_date='20141231',
                                                      min_lat=-10, max_lat=10, pressure=1)

    # get values at 10 hPa
    o3_10_hpa = np.zeros(len(o3.time))
    for i in range(len(o3.time)):
        pressure_i = pres_o3[i, :]
        o3_i = o3[i, :]
        n, l = helper_functions.interpolate_to_mls_pressure(pressure_i, o3_i)
        # only keep vmr at 10 hPa
        o3_10_hpa[i] = n[np.where(l == 10)]

    o3_10_hpa_dataset = xr.DataArray(o3_10_hpa, coords=[o3.time], dims=["time"])
    monthlymeans = o3_10_hpa_dataset.groupby('time.month').mean('time')
    anomalies_o3 = o3_10_hpa_dataset.groupby('time.month') - monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})

    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(8, 10))

    ax1.plot([anomalies_o3.time.values[0], anomalies_o3.time.values[-1]], [0, 0], sns.xkcd_rgb["light grey"])
    ax1.plot(anomalies_o3.time, anomalies_o3, sns.xkcd_rgb["hot pink"], marker='o',
             label="OSIRIS O$\mathregular{_3}$ [ppmv]", markersize=5)
    ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    ax1.set_ylabel('Anomaly')

    ax2.plot([anomalies_nox.time.values[0], anomalies_nox.time.values[-1]], [0, 0], sns.xkcd_rgb["light grey"])
    ax2.plot(anomalies_nox.time, anomalies_nox, sns.xkcd_rgb["deep sky blue"], marker='o',
             label='OSIRIS NO$\mathregular{_x}$ [ppbv]', markersize=5)
    ax2.plot(anomalies_hno3.Time, anomalies_hno3, sns.xkcd_rgb["indigo"], marker='D',
             label='MLS HNO$\mathregular{_3}$ [ppbv]', markersize=5)
    ax2.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    ax2.set_ylabel('Anomaly')

    ax3.plot([anomalies_n2o.Time.values[0], anomalies_n2o.Time.values[-1]], [0, 0], sns.xkcd_rgb["light grey"])
    ax3.plot(anomalies_n2o.Time, anomalies_n2o, sns.xkcd_rgb["vivid purple"], marker='D',
             label='MLS N$\mathregular{_2}$O [ppbv]', markersize=5)
    ax3.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
               ncol=3, mode="expand", borderaxespad=0.)
    ax3.set_ylabel('Anomaly')

    plt.xlabel('')
    ax1.set_ylim([-1.0, 1.0])
    ax2.set_ylim([-2.0, 2.0])
    ax3.set_ylim([-40, 40])
    plt.tight_layout(rect=[0, 0, 1, 0.98], h_pad=2.0)
    # plt.savefig("/home/kimberlee/Masters/NO2/Figures/timeseries_10hPa_32km.png", format='png', dpi=150)
    plt.show()

