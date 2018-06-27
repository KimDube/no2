"""
Open data from SAGE II, OSIRIS, MLS
"""

import xarray as xr
import pandas as pd
import numpy as np
from pysagereader.sage_ii_reader import SAGEIILoaderV700


# # # # # #
# SAGE II #
# # # # # #

def load_sage_ii_no2(start_date='1984-10-24', end_date='2005-8-31', min_lat=-10, max_lat=10, sun='set'):
    """
    :param start_date: string 'yyyy-m-d' in range 1984-10-24 to 2005-8-31.
    :param end_date: string 'yyyy-m-d' in range 1984-10-24 to 2005-8-31.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :param sun: NO2 values to return: string, one of [rise, set, both]
    :return: xarray of monthly mean SAGE II NO2 in time and latitude range with bad NO2 values removed. Measurements
    from either sunrise, sunset, or return both.
    """
    assert (sun == 'rise') or (sun == 'set') or (sun == 'both'), "sun must be one of [rise, set, both]"

    sage = SAGEIILoaderV700()
    sage.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage.load_data(start_date, end_date, min_lat, max_lat)
    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    if sun == 'set':
        data['NO2'][data['Type_Tan'] == 0] = np.nan  # remove sunrise events
        sunset = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2'])}, coords={'altitude': no2_alts, 'time': dates})
        sunset = sunset.resample(time='MS').mean('time', skipna=True)
        return sunset
    elif sun == 'rise':
        data['NO2'][data['Type_Tan'] == 1] = np.nan  # remove sunset events
        sunrise = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2'])}, coords={'altitude': no2_alts, 'time': dates})
        sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
        return sunrise
    elif sun == 'both':
        no2 = np.copy(data['NO2'])
        no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
        sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2)}, coords={'altitude': no2_alts, 'time': dates})
        sunset = sunset.resample(time='MS').mean('time', skipna=True)
        data['NO2'][data['Type_Tan'] == 1] = np.nan  # remove sunset events
        sunrise = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2'])}, coords={'altitude': no2_alts, 'time': dates})
        sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
        return sunrise, sunset


# # # # # #
# OSIRIS  #
# # # # # #

def load_osiris_nox_monthly(start_date='20050101', end_date='20141231', min_lat=-10, max_lat=10, pressure=0):
    """
    :param start_date: string 'yyyymmdd' in range 2004-2015. Default is '20050101'.
    :param end_date: string 'yyyymmdd' in range 2004-2015, greater than start_date. Default is '20141231'.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :param pressure: if != 0 return monthly mean pressure in addition to NOx vmr.
    :return: monthly mean OSIRIS NOx VMR in ppbv.
    """
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice(start_date, end_date))

    nox = datafile.derived_daily_mean_NOx_concentration.where(
        (datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    nox *= 6.022140857e17
    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    temp = datafile.temperature.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
    # Monthly mean
    nox = nox.resample('MS', dim='time', how='mean')
    if pressure != 0:
        pres = pres.resample('MS', dim='time', how='mean')
        return nox, pres
    else:
        return nox


def load_osiris_ozone_monthly(start_date='20050101', end_date='20141231', min_lat=-10, max_lat=10, pressure=0):
    """
    :param start_date: string 'yyyymmdd' in range 2004-2015. Default is '20050101'.
    :param end_date: string 'yyyymmdd' in range 2004-2015, greater than start_date. Default is '20141231'.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :param pressure: if != 0 return monthly mean pressure in addition to O3 vmr.
    :return: monthly mean OSIRIS O3 VMR in ppmv.
    """
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/CCI/OSIRIS_v5_10/*.nc')
    datafile = datafile.sel(time=slice(start_date, end_date))
    datafile = datafile.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    o3 = datafile.ozone_concentration * 6.022140857e17
    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    temp = datafile.temperature.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    o3 = (o3 * temp * 1.3806503e-19 / pres) * 1e6  # ppmv
    # Monthly mean
    o3 = o3.resample('MS', dim='time', how='mean')
    if pressure != 0:
        pres = pres.resample('MS', dim='time', how='mean')
        return o3, pres
    else:
        return o3


# # # #
# MLS #
# # # #

def read_mls_hno3(year='2005', day='0*', min_lat=-10, max_lat=10):
    """
    Load several days of MLS HNO3 data...files are large. A better computer might be able to handle a year.
     Filter to input latitude band, remove low quality data (re. MLS documentation).
     **** Monthly mean is not calculated cause full months are not necessarily loaded. ***
    :param year: string. Default is '2005'.
    :param day: string. Default is '0*'  which loads data for days 1 to 99 of year.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :return: Save a new netcdf file that is smaller and quicker to load.
    """
    hno3 = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_%sd%s.he5' % (year, day),
        group='HDFEOS/SWATHS/HNO3/Data Fields/', concat_dim='nTimes')
    geo = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_%sd%s.he5' % (year, day),
        group='HDFEOS/SWATHS/HNO3/Geolocation Fields/', concat_dim='nTimes')

    mls = xr.merge([hno3, geo])
    mls = mls.drop(['AscDescMode', 'L2gpPrecision', 'L2gpValue', 'ChunkNumber', 'LineOfSightAngle',
                    'LocalSolarTime', 'OrbitGeodeticAngle', 'SolarZenithAngle'])

    mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
    mls = xr.decode_cf(mls)
    mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
    mls = mls.dropna(dim="Time")
    mls = mls.where((mls.Latitude > min_lat) & (mls.Latitude < max_lat))

    # Lower stratosphere
    mls_l = mls.where(mls.Pressure > 22)
    mls_l = mls_l.where(((mls_l.Status % 2) == 0) & (mls_l.Quality > 0.8) & (mls_l.Convergence < 1.03))

    # Upper stratosphere
    mls_u = mls.where(mls.Pressure < 22)
    mls_u = mls_u.where((mls_u.Status != 0) & (mls_u.Quality > 0.8) & (mls_u.Convergence < 1.4))

    mls = xr.merge([mls_u, mls_l])

    mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/quarters/MLS-HNO3-%s-%s.nc' %
                       (year, day), mode='w')
    return


def combine_mls(year='2005', dataset='HNO3'):
    """
    Merge together a year of MLS HNO3 or O3 files (ideally from read_mls_(hn)o3 otherwise they will be too big to save)
    :param year: String. Default is '2005'. Combine files with this year in the extension and take monthly mean.
    :param dataset: String. Either 'HNO3' (default) or 'O3', otherwise the files won't be found.
    :return: Save new file with monthly means from input year.
    """
    if (dataset != 'HNO3') or (dataset != 'O3'):
        print('Dataset must be one of [HNO3, O3]')
        return
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_%s_monthlymeans/quarters/MLS-%s-%s-*.nc' %
                                 (dataset, dataset, year))
    mls = datafile.resample('MS', dim='Time', how='mean')
    mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_%s_monthlymeans/MLS-%s-%s.nc' %
                       (dataset, dataset, year), mode='w')
    return


def read_mls_n2o(year='2005', min_lat=-10, max_lat=10):
    """
    Load a year worth of MLS N2O data. Filter to input latitude band, remove low quality data (re. MLS documentation)
     and calculate monthly mean.
    :param year: string. Default is '2005'.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :return: Save a new netcdf file that is smaller and quicker to load.
    """
    n2o = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2N2Ov-04-23/MLS-Aura_L2GP-N2O_v04-20-c01_%s*.he5' % year,
        group='HDFEOS/SWATHS/N2O/Data Fields/', concat_dim='nTimes')
    geo = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2N2Ov-04-23/MLS-Aura_L2GP-N2O_v04-20-c01_%s*.he5' % year,
        group='HDFEOS/SWATHS/N2O/Geolocation Fields/', concat_dim='nTimes')

    mls = xr.merge([n2o, geo])
    mls = mls.drop(['AscDescMode', 'L2gpPrecision', 'L2gpValue', 'ChunkNumber', 'LineOfSightAngle',
                    'LocalSolarTime', 'OrbitGeodeticAngle', 'SolarZenithAngle'])
    mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
    mls = xr.decode_cf(mls)
    mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
    mls = mls.dropna(dim="Time")
    mls = mls.where((mls.Latitude > min_lat) & (mls.Latitude < max_lat))
    mls = mls.where((mls.Status % 2) == 0)
    mls = mls.where(mls.Quality > 1.4)
    mls = mls.where(mls.Convergence < 1.01)
    mls = mls.resample('MS', dim='Time', how='mean')

    mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-%s.nc' % year, mode='w')
    return


def read_mls_o3(year='2005', day='0*', min_lat=-10, max_lat=10):
    """
    Load several days of MLS O3 data...files are large. A better computer might be able to handle a year.
     Filter to input latitude band, remove low quality data (re. MLS documentation).
     **** Monthly mean is not calculated cause full months are not necessarily loaded. ***
    :param year: string. Default is '2005'.
    :param day: string. Default is '0*'  which loads data for days 1 to 99 of year.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :return: Save a new netcdf file that is smaller and quicker to load.
    """
    o3 = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2O3v04-20/MLS-Aura_L2GP-O3_v04-20-c01_%sd%s.he5' % (year, day),
        group='HDFEOS/SWATHS/O3/Data Fields/', concat_dim='nTimes')
    geo = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2O3v04-20/MLS-Aura_L2GP-O3_v04-20-c01_%sd%s.he5' % (year, day),
        group='HDFEOS/SWATHS/O3/Geolocation Fields/', concat_dim='nTimes')

    mls = xr.merge([o3, geo])
    mls = mls.drop(['AscDescMode', 'L2gpPrecision', 'L2gpValue', 'ChunkNumber', 'LineOfSightAngle', 'LocalSolarTime',
                    'OrbitGeodeticAngle', 'SolarZenithAngle'])

    mls.Time.attrs['units'] = 'Seconds since 01-01-1993'
    mls = xr.decode_cf(mls)
    mls = mls.swap_dims({'nTimes': 'Time'}, inplace=True)
    mls = mls.dropna(dim="Time")
    mls = mls.where((mls.Latitude > min_lat) & (mls.Latitude < max_lat))
    mls = mls.where(((mls.Status % 2) == 0) & (mls.Quality > 1.0) & (mls.Convergence < 1.03) & (mls.O3Precision > 0))
    mls = mls.resample('MS', dim='Time', how='mean')

    mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/quarters/MLS-O3-%s-%s.nc' %
                       (year, day), mode='w')
    return


if __name__ == "__main__":
    read_mls_n2o(year='2015', min_lat=-10, max_lat=10)

