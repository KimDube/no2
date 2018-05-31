###
# Load MLS data and take monthly means
###

import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr


# Open files.
n2o = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2O3v04-20/MLS-Aura_L2GP-O3_v04-20-c01_2009d3*.he5',
                      group='HDFEOS/SWATHS/O3/Data Fields/', concat_dim='nTimes')
geo = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2O3v04-20/MLS-Aura_L2GP-O3_v04-20-c01_2009d3*.he5',
                      group='HDFEOS/SWATHS/O3/Geolocation Fields/', concat_dim='nTimes')

mls = xr.merge([n2o, geo])

mls = mls.drop(['AscDescMode', 'L2gpPrecision', 'L2gpValue', 'ChunkNumber', 'LineOfSightAngle', 'LocalSolarTime',
                'OrbitGeodeticAngle', 'SolarZenithAngle'])

mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
mls = xr.decode_cf(mls)
mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
mls = mls.dropna(dim="Time")
mls = mls.where((mls.Latitude > -10) & (mls.Latitude < 10))

mls = mls.where(((mls.Status % 2) == 0) & (mls.Quality > 1.0) & (mls.Convergence < 1.03) & (mls.O3Precision > 0))
mls = mls.resample('MS', dim='Time', how='mean')

mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/quarters/MLS-O3-2009-3.nc', mode='w')


