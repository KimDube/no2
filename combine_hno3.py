import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


# datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/quarters/MLS-HNO3-2014-*.nc')
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/quarters/MLS-O3-2010-*.nc')
# datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/MLS-O3-2012.nc')

mls = datafile.resample('MS', dim='Time', how='mean')

print(mls)

# mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-2014.nc', mode='w')
mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/MLS-O3-2010.nc', mode='w')


"""
da = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/MLS-O3-2010.nc')

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(10, 5))
fax = da.O3.plot.contourf(x='Time', y='nLevels', robust=True, cmap="RdBu_r", extend='both', add_colorbar=0)

plt.show()
"""