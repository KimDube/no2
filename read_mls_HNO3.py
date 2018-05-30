###
# Load MLS data and take monthly means
###

import xarray as xr


# Open files.
n2o = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_2014d0*.he5',
    group='HDFEOS/SWATHS/HNO3/Data Fields/', concat_dim='nTimes')

geo = xr.open_mfdataset(
    r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_2014d0*.he5',
    group='HDFEOS/SWATHS/HNO3/Geolocation Fields/', concat_dim='nTimes')

mls = xr.merge([n2o, geo])

mls = mls.drop(['AscDescMode', 'L2gpPrecision', 'L2gpValue', 'ChunkNumber', 'LineOfSightAngle', 'LocalSolarTime',
                'OrbitGeodeticAngle', 'SolarZenithAngle'])

mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
mls = xr.decode_cf(mls)
mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
mls = mls.dropna(dim="Time")
mls = mls.where((mls.Latitude > -10) & (mls.Latitude < 10))

# Lower stratosphere
mls_l = mls.where(mls.Pressure > 22)
mls_l = mls_l.where(((mls_l.Status % 2) == 0) & (mls_l.Quality > 0.8) & (mls_l.Convergence < 1.03))

# Upper stratosphere
mls_u = mls.where(mls.Pressure < 22)
mls_u = mls_u.where((mls_u.Status != 0) & (mls_u.Quality > 0.8) & (mls_u.Convergence < 1.4))

mls = xr.merge([mls_u, mls_l])
# mls = mls.resample('MS', dim='Time', how='mean')

mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/quarters/MLS-HNO3-2014-0.nc', mode='w')
