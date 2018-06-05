import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt

# Load NOx
dataf = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
dataf = dataf.swap_dims({'profile_id': 'time'}, inplace=True)
dataf = dataf.sel(time=slice('20050101', '20051231'))
nox = dataf.derived_0630_NOx_concentration.where((dataf.latitude > -10) & (dataf.latitude < 10))
lats = dataf.latitude.where((dataf.latitude > -10) & (dataf.latitude < 10))
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
nox *= 6.022140857e17
# To convert number density to vmr
pres = dataf.pressure.where((dataf.latitude > -10) & (dataf.latitude < 10))
temp = dataf.temperature.where((dataf.latitude > -10) & (dataf.latitude < 10))
nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv

nox = nox.resample('MS', dim='time', how='mean')

print(nox)

nox.to_netcdf(path='/home/kimberlee/Masters/NO2/OSIRIS_NOx_monthlymeans/OSIRIS-NOx-2005.nc', mode='w')

# =====================================================

da = xr.open_mfdataset('/home/kimberlee/Masters/NO2/OSIRIS_NOx_monthlymeans/OSIRIS-NOx-2005.nc')

print(da)

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(10, 5))
fax = plt.contourf(da['time'], da['altitude'], da, robust=True, cmap="RdBu_r", extend='both', add_colorbar=0)

plt.show()