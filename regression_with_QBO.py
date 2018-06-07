###
# Fit QBO wind components to NOx
###


import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA


# Need PCA of Singapore winds... old code has disappeared
sklearn_pca = PCA(n_components=2)

# Load NOx
datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
datafile = datafile.sel(time=slice('20050101', '20141231'))

nox = datafile.derived_daily_mean_NOx_concentration.where((datafile.latitude > -5) & (datafile.latitude < 5))
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
nox *= 6.022140857e17
# To convert number density to vmr
pres = datafile.pressure.where((datafile.latitude > -5) & (datafile.latitude < 5))
temp = datafile.temperature.where((datafile.latitude > -5) & (datafile.latitude < 5))
nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv

nox = nox.resample('MS', dim='time', how='mean')
monthlymeans = nox.groupby('time.month').mean('time')
anomalies = nox.groupby('time.month') - monthlymeans
