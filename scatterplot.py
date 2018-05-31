###
# Plot N2O vs HNO3, O3, NOx
###

import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns


# Load N2O
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20141231'))
nox = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
monthlymeans = nox.groupby('Time.month').mean('Time')
anomalies_n2o = nox.groupby('Time.month') - monthlymeans
# anomalies_n2o['nLevels'] = datafile.Pressure.mean('Time')
anomalies_n2o *= 1e9  # ppbv
anomalies_n2o = anomalies_n2o.sel(nLevels=11)  # 10 hPa

# Load HNO3
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20141231'))
nox = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
monthlymeans = nox.groupby('Time.month').mean('Time')
anomalies_hno3 = nox.groupby('Time.month') - monthlymeans
# anomalies_hno3['nLevels'] = datafile.Pressure.mean('Time')
anomalies_hno3 *= 1e9  # ppbv
anomalies_hno3 = anomalies_hno3.sel(nLevels=11)  # 10 hPa

# Load NOx
dataf = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
dataf = dataf.swap_dims({'profile_id': 'time'}, inplace=True)
dataf = dataf.sel(time=slice('20050101', '20141231'))
nox = dataf.derived_0630_NOx_concentration.where((dataf.latitude > -10) & (dataf.latitude < 10))
nox = nox.sel(altitude=26.5)
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
nox *= 6.022140857e17
# To convert number density to vmr
pres = dataf.pressure.where((dataf.latitude > -10) & (dataf.latitude < 10))
temp = dataf.temperature.where((dataf.latitude > -10) & (dataf.latitude < 10))
nox = (nox * temp.sel(altitude=26.5) * 1.3806503e-19 / pres.sel(altitude=26.5)) * 1e9  # ppbv
nox = nox.resample('MS', dim='time', how='mean')
monthlymeans = nox.groupby('time.month').mean('time')
anomalies_nox = nox.groupby('time.month') - monthlymeans

# Load O3
datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/CCI/OSIRIS_v5_10/*.nc')
datafile = datafile.sel(time=slice('20050101', '20141231'))
o3 = datafile.ozone_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
o3 = o3.sel(altitude=26.)
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
o3 *= 6.022140857e17
# To convert number density to vmr
pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))
o3 = (o3 * temp.sel(altitude=26.) * 1.3806503e-19 / pres.sel(altitude=26.)) * 1e6  # ppmv
o3 = o3.resample('MS', dim='time', how='mean')
monthlymeans = o3.groupby('time.month').mean('time')
anomalies_o3 = o3.groupby('time.month') - monthlymeans


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