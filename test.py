
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dataf = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/CCI/OSIRIS_v5_10/*.nc')
#dataf = dataf.swap_dims({'profile_id': 'time'}, inplace=True)
dataf = dataf.sel(time=slice('20080301', '20080331'))  # one month
dataf = dataf.mean('altitude')
# dataf = dataf.resample('MS', dim='time', how='mean')

dataf['ozone_concentration'] *= 6.022140857e17
lats = np.arange(-60, 70, 10)
longs = np.arange(-180, 190, 10)

map = np.zeros((len(lats), len(longs)))

for i in range(len(lats)):
    for j in range(len(longs)):
        mmm = dataf.ozone_concentration.where((dataf.latitude > i) & (dataf.latitude < i+10) &
                                (dataf.longitude > j) & (dataf.longitude < j+10))
        map[i, j] = np.nanmean(mmm)

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(12, 6))
im = plt.contourf(longs, lats, map, np.arange(1.2e12, 1.5e12, 1e10))
cb = plt.colorbar(im, fraction=0.05, pad=0.02)
cb.set_label("Ozone Number Density")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

exit()

o3 = dataf.where((dataf.latitude > -10) & (dataf.latitude < 10))
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
o3 *= 6.022140857e17
# To convert number density to vmr
#pres = dataf.pressure.where((dataf.latitude > -10) & (dataf.latitude < 10))
#temp = dataf.temperature.where((dataf.latitude > -10) & (dataf.latitude < 10))
o3 = (o3 * o3.temperature * 1.3806503e-19 / o3.pressure)
nox = o3.resample('MS', dim='time', how='mean')

"""
sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 8))
plt.title("Mean N2O - MLS")
plt.xlabel("N2O [VMR]")
plt.ylabel("Pressure [hPa]")
plt.plot(nox.ozone_concentration[4, :]*1e9, nox.altitude, 'o')
plt.gca().invert_yaxis()
plt.show()
"""

nox.to_netcdf(path='/home/kimberlee/Masters/NO2/OSIRIS_NOx_monthlymeans/OSIRIS-NOx-2006.nc', mode='w')

datafile = xr.open_dataset('/home/kimberlee/Masters/NO2/OSIRIS_NOx_monthlymeans/OSIRIS-NOx-2006.nc')
sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['purple']
sns.set_palette(sns.xkcd_palette(colours))
fig, ax = plt.subplots(figsize=(8, 8))
plt.title("Mean N2O - MLS")
plt.xlabel("N2O [VMR]")
plt.ylabel("Pressure [hPa]")
plt.semilogy(datafile.ozone_concentration*1e9, datafile.altitude, 'o')
plt.gca().invert_yaxis()
plt.show()