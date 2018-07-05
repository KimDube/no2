
# # #
# Plot correlation coefficient as a function of altitude and latitude for SAGE II and OSIRIS NO2
# # #

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pysagereader.sage_ii_reader import SAGEIILoaderV700
import helper_functions
import open_data
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

    assert (sage_type == 'sunrise') or (sage_type == 'sunset') or (sage_type == 'both'), \
        "sage_type must be one of [sunrise, sunset, both]"
    assert (osiris_type == '630') or (osiris_type == 'dailymean') or (osiris_type == 'measured'), \
        "osiris_type must be one of [630, dailymean, measured]"

    cc = np.zeros((len(lats)-1, len(alts)-1))  # return correlation coeffs

    # load sage II
    sage_loader = SAGEIILoaderV700()
    sage_loader.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage_loader.load_data('1996-1-1', '2005-8-31', lats[0], lats[-1])
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
    elif sage_type == 'both':
        no2 = np.copy(data['NO2'])
        no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
        sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2), 'latitude': (['time'], data['Lat'])},
                              coords={'altitude': no2_alts, 'time': dates})
        sunset = sunset.sel(altitude=slice(alts[0], alts[-1]))
        data['NO2'][data['Type_Tan'] == 1] = np.nan  # remove sunset events
        sunrise = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2']), 'latitude': (['time'], data['Lat'])},
                              coords={'altitude': no2_alts, 'time': dates})
        sunrise = sunrise.sel(altitude=slice(alts[0], alts[-1]))

    # load OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20161231'))

    if osiris_type == 'dailymean':
        no2_osiris = datafile.derived_daily_mean_NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))
    elif osiris_type == '630':
        no2_osiris = datafile.derived_0630_NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))
    elif osiris_type == 'measured':
        no2_osiris = datafile.NO2_concentration.where(
            (datafile.latitude > lats[0]) & (datafile.latitude < lats[-1]))
        no2_osiris = no2_osiris.sel(altitude=slice(alts[0], alts[-1]))

    for i in range(len(lats) - 1):  # get standardized anomalies in latitude band
        print(lats[i])
        if sage_type == 'both':
            sunset_lat = sunset.where((sunset.latitude > lats[i]) & (sunset.latitude < lats[i + 1]))
            sunrise_lat = sunrise.where((sunrise.latitude > lats[i]) & (sunrise.latitude < lats[i + 1]))
            sunset_lat = sunset_lat.resample(time='MS').mean('time', skipna=True)
            sunrise_lat = sunrise_lat.resample(time='MS').mean('time', skipna=True)
            sunset_anom = helper_functions.relative_anomaly(sunset_lat)
            sunrise_anom = helper_functions.relative_anomaly(sunrise_lat)
            no2_sage_anom = open_data.sage_ii_no2_combine_sunrise_sunset(sunrise_anom, sunset_anom, alts[0], alts[-1])
            no2_sage_anom = xr.Dataset({'NO2': (['time', 'altitude'], no2_sage_anom)},
                              coords={'altitude': sunset_anom.altitude, 'time': sunset_anom.time})
            no2_sage_anom = no2_sage_anom.where((no2_sage_anom.altitude % 1) == 0.5, drop=True)
        else:
            no2_sage_lat = no2_sage.where((no2_sage.latitude > lats[i]) & (no2_sage.latitude < lats[i + 1]))
            no2_sage_lat = no2_sage_lat.resample(time='MS').mean('time', skipna=True)
            no2_sage_anom = helper_functions.relative_anomaly(no2_sage_lat)

        no2_osiris_lat = no2_osiris.where((no2_osiris.latitude > lats[i]) & (no2_osiris.latitude < lats[i + 1]))
        no2_osiris_lat = no2_osiris_lat.resample(time='MS').mean('time', skipna=True)
        no2_osiris_anom = helper_functions.relative_anomaly(no2_osiris_lat)

        # Select data from overlap period only now that anomaly has been found.
        no2_sage_anom = no2_sage_anom.sel(time=slice('2002-1-1', '2005-8-31'))
        no2_osiris_anom = no2_osiris_anom.sel(time=slice('20020101', '20050831'))

        for j in range(len(alts) - 1):  # find correlation coefficient at each altitude
            with warnings.catch_warnings():  # invalid value encountered in true_divide
                warnings.simplefilter("ignore", category=RuntimeWarning)
                no2_osiris_at_alt = no2_osiris_anom.where(no2_osiris_anom.altitude == alts[j], drop=True)
                no2_sage_at_alt = no2_sage_anom.where(no2_sage_anom.altitude == alts[j], drop=True)

                # Interpolate missing values, otherwise correlation won't work
                no2_osiris_at_alt = no2_osiris_at_alt.chunk({'time': -1})
                no2_osiris_at_alt = no2_osiris_at_alt.interpolate_na(dim='time', method='linear')
                no2_sage_at_alt = no2_sage_at_alt.NO2.interpolate_na(dim='time', method='linear')

                # no2_osiris_at_alt = helper_functions.get_median_filtered(no2_osiris_at_alt.values)

                cc[i, j], p = pearsonr(no2_osiris_at_alt[1:-5], no2_sage_at_alt[1:-4])  # sunset
                # cc[i, j], p = pearsonr(no2_osiris_at_alt[2:-5], no2_sage_at_alt[2:-4])  # sunrise
    return cc


if __name__ == "__main__":
    latitudes = np.arange(-55, 65, 10)
    altitudes = np.arange(24.5, 40.5)

    corrcoeffs = correlate_sage_osiris(latitudes, altitudes, 'sunset', '630')

    plotlats = np.arange(-50, 60, 10)

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(12, 8))
    im = sns.heatmap(corrcoeffs.T, center=0, annot=True, cmap='RdYlBu_r', xticklabels=plotlats,
                     yticklabels=altitudes[0:-1],
                     cbar_kws={'label': 'Correlation'})
    im.set_yticklabels(im.get_yticklabels(), rotation=0)
    ax.invert_yaxis()
    plt.ylabel("Altitude [km]")
    plt.xlabel("Latitude")
    plt.title("OSIRIS 630 AM & SAGE Sunset")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_osiris_corr_sunset_630.png", format='png', dpi=150)
    plt.show()
