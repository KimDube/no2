
# # #
# Plot correlation coefficient as a function of altitude and latitude for SAGE II and OSIRIS NO2
# # #

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pysagereader.sage_ii_reader import SAGEIILoaderV700
import xarray as xr
from scipy.stats.stats import pearsonr
import warnings


def correlate_sage_osiris(lats, alts, sage_type, osiris_type):
    """
    Correlate osiris and sage NO2 as a function of altitude and latitude for overlap period, 2002-20050831.
    :param lats: array of latitude bins
    :param alts: array of altitude ending in 0.5
    :param sage_type: sage measurements to use: sunrise or sunset
    :param osiris_type: osiris data to used: 630 (derived), dailymean (derived), measured
    :return: cc - array of correlation coefficients [latitude, altitude]
    """

    assert (sage_type == 'sunrise') or (sage_type == 'sunset'), "sage_type must be one of [sunrise, sunset]"
    assert (osiris_type == '630') or (osiris_type == 'dailymean') or (osiris_type == 'measured'), \
        "osiris_type must be one of [630, dailymean, measured]"

    cc = np.zeros((len(lats), len(alts)))  # return correlation coeffs

    # load sage II
    sage_loader = SAGEIILoaderV700()
    sage_loader.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage_loader.load_data('2002-1-1', '2005-8-31', lats[0], lats[-1])
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))  # convert dates
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    if sage_type == 'sunset':
        data['NO2'][data['Type_Tan'] == 0] = np.nan  # remove sunrise events
        no2_sage = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2']), 'latitude': (['time'], data['Lat'])},
                              coords={'altitude': no2_alts, 'time': dates})
        no2_sage = no2_sage.sel(altitude=slice(alts[0], alts[-1]))
        no2_sage = no2_sage.where((no2_sage.altitude % 1) == 0.5, drop=True)  # every 2nd altitude to match OSIRIS

    elif sage_type == 'sunrise':
        data['NO2'][data['Type_Tan'] == 1] = np.nan  # remove sunset events
        no2_sage = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2']), 'latitude': (['time'], data['Lat'])},
                              coords={'altitude': no2_alts, 'time': dates})
        no2_sage = no2_sage.sel(altitude=slice(alts[0], alts[-1]))
        no2_sage = no2_sage.where((no2_sage.altitude % 1) == 0.5, drop=True)  # every 2nd altitude to match OSIRIS

    # load OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20050731'))

    if osiris_type == 'dailymean':
        no2_osiris = datafile.derived_daily_mean_NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))
        no2_osiris *= 6.022140857e17  # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    elif osiris_type == '630':
        no2_osiris = datafile.derived_0630_NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))
        no2_osiris *= 6.022140857e17  # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    elif osiris_type == 'measured':
        no2_osiris = datafile.NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))
        no2_osiris *= 6.022140857e17  # to convert concentration to number density [mol/m^3 to molecule/cm^3]

    for i in range(len(lats) - 1):  # get standardized anomalies in latitude band
        print(lats[i])
        no2_sage_lat = no2_sage.where((no2_sage.latitude > lats[i]) & (no2_sage.latitude < lats[i + 1]))
        no2_sage_lat = no2_sage_lat.resample(time='MS').mean('time', skipna=True)
        no2_sage_anom = no2_sage_lat.groupby('time.month') - no2_sage_lat.groupby('time.month').mean('time')
        no2_sage_anom = no2_sage_anom.groupby('time.month') / no2_sage_lat.groupby('time.month').mean('time')

        no2_osiris_lat = no2_osiris.where((no2_osiris.latitude > lats[i]) & (no2_osiris.latitude < lats[i + 1]))
        no2_osiris_lat = no2_osiris_lat.resample(time='MS').mean('time', skipna=True)
        no2_osiris_anom = no2_osiris_lat.groupby('time.month') - no2_osiris_lat.groupby('time.month').mean('time')
        no2_osiris_anom = no2_osiris_anom.groupby('time.month') / no2_osiris_lat.groupby('time.month').mean('time')

        for j in range(len(alts) - 1):  # find correlation coefficient at each altitude
            with warnings.catch_warnings():  # invalid value encountered in true_divide
                warnings.simplefilter("ignore", category=RuntimeWarning)
                no2_osiris_at_alt = no2_osiris_anom.where(no2_osiris_anom.altitude == alts[j], drop=True)
                no2_sage_at_alt = no2_sage_anom.where(no2_sage_anom.altitude == alts[j], drop=True)

                # Interpolate missing values, otherwise correlation won't work
                no2_osiris_at_alt = no2_osiris_at_alt.chunk({'time': -1})
                no2_osiris_at_alt = no2_osiris_at_alt.interpolate_na(dim='time', method='linear')
                no2_sage_at_alt = no2_sage_at_alt.NO2.interpolate_na(dim='time', method='linear')

                # cc[i, j], p = pearsonr(no2_osiris_at_alt[1:-4], no2_sage_at_alt[1:-4])  # sunset
                cc[i, j], p = pearsonr(no2_osiris_at_alt[2:-4], no2_sage_at_alt[2:-4])  # sunrise
    return cc


if __name__ == "__main__":
    latitudes = np.arange(-55, 65, 10)
    altitudes = np.arange(24.5, 40.5)

    corrcoeffs = correlate_sage_osiris(latitudes, altitudes, 'sunrise', 'dailymean')

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(8, 6))
    im = plt.pcolormesh(latitudes, altitudes, corrcoeffs.T, vmin=-1, vmax=1, cmap='RdYlBu_r')
    plt.ylabel("Altitude [km]")
    plt.xlabel("Latitude")
    plt.title("OSIRIS Daily Mean & SAGE Sunrise")
    cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
    cb.set_label("Correlation Coeff.")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_osiris_corr_sunrise_dailymean.png", format='png', dpi=150)
    plt.show()
