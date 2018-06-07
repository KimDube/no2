
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NO2 import alt_to_pres

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
    nox_10_hpa[i] = n[7]

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
    o3_10_hpa[i] = n2[7]

o3_10_hpa_dataset = xr.DataArray(o3_10_hpa, coords=[o3.time], dims=["time"])
monthlymeans = o3_10_hpa_dataset.groupby('time.month').mean('time')
anomalies_o3 = o3_10_hpa_dataset.groupby('time.month') - monthlymeans


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 7))
sns.regplot(y=anomalies_o3, x=anomalies_nox, color=sns.xkcd_rgb["bright orange"], ci=None,
            marker='D', scatter_kws={"s": 50})

ax.set_xlabel("OSIRIS NO$\mathregular{_x}$ (ppbv)")
ax.set_ylabel("OSIRIS O$\mathregular{_3}$ (ppmv)")
plt.title("10.0 hPa")
ax.set_xlim([-2.2, 2.2])
ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/NO2/Figures/NOx_v_O3_10hPa.png", format='png', dpi=150)
plt.show()