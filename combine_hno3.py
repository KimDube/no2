import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/quarters/MLS-HNO3-2014-*.nc')

mls = datafile.resample('MS', dim='Time', how='mean')

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(10, 5))
fax = mls.HNO3.plot.contourf(x='Time', y='nLevels', robust=True, cmap="RdBu_r", extend='both', add_colorbar=0)

plt.show()

mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-2014.nc', mode='w')
