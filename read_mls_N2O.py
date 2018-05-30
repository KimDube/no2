###
# Load MLS data and take monthly means
###

import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


# Open files.
n2o = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2N2Ov-04-23/MLS-Aura_L2GP-N2O_v04-20-c01_2014d*.he5',
                      group='HDFEOS/SWATHS/N2O/Data Fields/', concat_dim='nTimes')
geo = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2N2Ov-04-23/MLS-Aura_L2GP-N2O_v04-20-c01_2014d*.he5',
                      group='HDFEOS/SWATHS/N2O/Geolocation Fields/', concat_dim='nTimes')

mls = xr.merge([n2o, geo])
mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
mls = xr.decode_cf(mls)
mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
mls = mls.dropna(dim="Time")
mls = mls.where((mls.Latitude > -10) & (mls.Latitude < 10))
mls = mls.where((mls.Status % 2) == 0)
mls = mls.where(mls.Quality > 1.4)
mls = mls.where(mls.Convergence < 1.01)
mls = mls.resample('MS', dim='Time', how='mean')


mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-2014.nc', mode='w')

# datafile = xr.open_dataset('/home/kimberlee/Masters/NO2/MLS-N2O-2005.nc')
"""
sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
colours = ['purple']
sns.set_palette(sns.xkcd_palette(colours))
fig, ax = plt.subplots(figsize=(8, 8))
plt.title("Mean N2O - MLS")
plt.xlabel("N2O [VMR]")
plt.ylabel("Pressure [hPa]")
plt.semilogy(mls.N2O*1e9, mls.Pressure, 'o')
plt.ylim([0.46, 100])
plt.gca().invert_yaxis()
# plt.savefig("/home/kimberlee/Masters/NO2/Figures/MLS_N2O_profile.png", format='png', dpi=150)
plt.show()
"""

