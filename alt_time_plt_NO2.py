###
# NO2 - Number density over altitude and time
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


if __name__ == "__main__":
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20161231'))
    tropics = datafile.derived_daily_mean_NO2_concentration.where((datafile.latitude > -45) & (datafile.latitude < -35))
    tropics = tropics.resample('MS', dim='time', how='mean')
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    tropics *= 6.022140857e17
    tropics /= 1e9

    monthlymeans = tropics.groupby('time.month').mean('time')
    anomalies = tropics.groupby('time.month') - monthlymeans
    anomalies = anomalies.groupby('time.month') / monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    #fax = anomalies.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-0.3, 0.32, 0.02),
    #                              cmap="RdBu_r", extend='both', add_colorbar=0)

    fax = plt.pcolormesh(anomalies.time, anomalies.altitude, anomalies.T, vmin=-0.3, vmax=0.3, cmap="RdBu_r")
    plt.ylim([25, 40])
    plt.ylabel('Altitude [km]')
    plt.xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    plt.title('Monthly Mean OSIRIS NO$\mathregular{_2}$ (-10 to 10 deg. lat.)')
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    # cb.set_label("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    cb.set_label("NO$\mathregular{_2}$ Anomaly")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/OSIRIS_no2_anomaly_35_45.png", format='png', dpi=150)
    plt.show()

