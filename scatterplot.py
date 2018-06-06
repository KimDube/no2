###
# Plot N2O vs HNO3, O3, NOx
###

import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NO2 import alt_to_pres


# Load N2O
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20141231'))
nox = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
monthlymeans = nox.groupby('Time.month').mean('Time')
anomalies_n2o = nox.groupby('Time.month') - monthlymeans
anomalies_n2o *= 1e9  # ppbv
anomalies_n2o = anomalies_n2o.sel(nLevels=10)  # 10 = 21.5 hPa, 12 = 10 hPa

# Load HNO3
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20141231'))
nox = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
monthlymeans = nox.groupby('Time.month').mean('Time')
anomalies_hno3 = nox.groupby('Time.month') - monthlymeans
anomalies_hno3 *= 1e9  # ppbv
anomalies_hno3 = anomalies_hno3.sel(nLevels=10)  # 10 = 21.5 hPa, 12 = 10 hPa

# Load NOx
dataf = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
dataf = dataf.swap_dims({'profile_id': 'time'}, inplace=True)
dataf = dataf.sel(time=slice('20050101', '20141231'))
nox = dataf.derived_daily_mean_NOx_concentration.where((dataf.latitude > -10) & (dataf.latitude < 10))
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
nox *= 6.022140857e17
# To convert number density to vmr
pres = dataf.pressure.where((dataf.latitude > -10) & (dataf.latitude < 10))
temp = dataf.temperature.where((dataf.latitude > -10) & (dataf.latitude < 10))
nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
nox = nox.resample('MS', dim='time', how='mean')
pres = pres.resample('MS', dim='time', how='mean')

# get values at 10 hPa
nox_10_hpa = np.zeros(len(nox.time))
for i in range(len(nox.time)):
    pressure_i = pres[i, :]
    nox_i = nox[i, :]
    n, l = alt_to_pres.interpolate_to_mls_pressure(pressure_i, nox_i)
    # l = 10 hPa at index 7, l = 21.5 hPa at index 5
    nox_10_hpa[i] = n[5]

nox_10_hpa_dataset = xr.DataArray(nox_10_hpa, coords=[nox.time], dims=["time"])
monthlymeans = nox_10_hpa_dataset.groupby('time.month').mean('time')
anomalies_nox = nox_10_hpa_dataset.groupby('time.month') - monthlymeans

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
pres = pres.resample('MS', dim='time', how='mean')

# get values at 10 hPa
o3_10_hpa = np.zeros(len(o3.time))
for i in range(len(nox.time)):
    pressure_i = pres[i, :]
    o3_i = o3[i, :]
    n2, l2 = alt_to_pres.interpolate_to_mls_pressure(pressure_i, o3_i)
    # l = 10 hPa at index 7, l = 21.5 hPa at index 5
    o3_10_hpa[i] = n2[5]

o3_10_hpa_dataset = xr.DataArray(o3_10_hpa, coords=[o3.time], dims=["time"])
monthlymeans = o3_10_hpa_dataset.groupby('time.month').mean('time')
anomalies_o3 = o3_10_hpa_dataset.groupby('time.month') - monthlymeans


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 7))
sns.regplot(x=anomalies_n2o, y=anomalies_hno3, color=sns.xkcd_rgb["bright orange"], ci=None,
            marker='D', scatter_kws={"s": 50}, label="HNO$\mathregular{_3}$")
sns.regplot(x=anomalies_n2o[0:-3], y=anomalies_o3, color=sns.xkcd_rgb["indigo"], ci=None,
            marker='<', scatter_kws={"s": 50}, label="O$\mathregular{_3}$")
ax2 = ax.twinx()
sns.regplot(x=anomalies_n2o[0:-3], y=anomalies_nox, color=sns.xkcd_rgb["azure"], ci=None,
            ax=ax2, marker='o', scatter_kws={"s": 50}, label="NO$\mathregular{_x}$")

ax.set_xlabel("MLS N$\mathregular{_2}$O (ppbv)")
ax2.set_ylabel("OSIRIS NO$\mathregular{_x}$ (ppbv)")
ax.set_ylabel("MLS HNO$\mathregular{_3}$ (ppbv) & OSIRIS O$\mathregular{_3}$ (ppmv)")
plt.title("21.5 hPa")
ax.set_xlim([-40, 40])

ax.legend(loc=2)
ax2.legend(bbox_to_anchor=(0.22, 0, 1., 1.0), loc=2)

plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/NO2/Figures/scatterplot_21hPa.png", format='png', dpi=150)
plt.show()