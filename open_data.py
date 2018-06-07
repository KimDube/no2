"""
Open data
"""

import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt


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

def read_mls_hno3(time='2005d0*', min_lat=-10, max_lat=10):
    """
    Load several days of MLS HNO3 data...files are large. A better computer might be able to handle a year.
     Filter to input latitude band, remove low quality data (re. MLS documentation).
     **** Monthly mean is not calculated cause full months are not necessarily loaded. ***
    :param time: string. Default is '2005d0*' which loads data for days 1 to 99 of 2005.
    :param min_lat: minimum latitude to include in means. Default is -10.
    :param max_lat: maximum latitude to include in means. Default is 10.
    :return: Save a new netcdf file that is smaller and quicker to load.
    """
    hno3 = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_%s.he5' % time,
        group='HDFEOS/SWATHS/HNO3/Data Fields/', concat_dim='nTimes')
    geo = xr.open_mfdataset(
        r'/home/kimberlee/ValhallaData/MLS/L2HNO3-v04-23/MLS-Aura_L2GP-HNO3_v04-20-c01_%s.he5' % time,
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

    mls.to_netcdf(path='/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/quarters/MLS-HNO3-2014-0.nc', mode='w')
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


def load_mls_n2o_monthly():

    return 0


if __name__ == "__main__":
    """
    read_mls_n2o('2014')

    mls = xr.open_dataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-2014.nc')

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
