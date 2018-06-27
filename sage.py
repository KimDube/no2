
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pysagereader.sage_ii_reader import SAGEIILoaderV700
import xarray as xr
import open_data


def sage_ii_ozone(data_folder):
    # load the data
    sage = SAGEIILoaderV700()
    sage.data_folder = data_folder
    data = sage.load_data('2002-1-1', '2002-12-31', -10, 10)

    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))

    # get rid of bad data
    data['O3'][data['O3'] == data['FillVal']] = np.nan
    data['O3'][data['O3_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    data['O3'][data['Type_Tan'] == 1] = np.nan  # remove sunrise events

    # get no2 altitudes
    o3_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_O3'][0]) & (data['Alt_Grid'] <= data['Range_O3'][1])]

    # Create dataset
    ds = xr.Dataset({'O3': (['time', 'altitude'], data['O3'])},
                    coords={'altitude': o3_alts,
                            'time': dates})

    ds = ds.sel(altitude=slice(24.5, 40.5))  # data above and below these altitudes is noisy
    monthly_o3 = ds.resample(time='MS').mean('time')
    monthlymeans = monthly_o3.groupby('time.month').mean('time')
    anomalies = monthly_o3.groupby('time.month') - monthlymeans
    anomalies = anomalies.groupby('time.month') / monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    # monthly_o3 /= 1e9
    fax = anomalies.O3.T.plot.contourf(x='time', y='altitude', robust=True,
                                          extend='both', add_colorbar=0, cmap="hot")
    plt.ylabel('Altitude [km]')
    plt.ylim(25, 40)
    plt.xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    # plt.title('Monthly Mean SAGE II O$\mathregular{_3}$ (-10 to 10 deg. lat.)')
    plt.title('Monthly Mean SAGE II O$\mathregular{_3}$ - Sunset (-10 to 10 deg. lat.)')
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    # cb.set_label("O$\mathregular{_3}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    cb.set_label("O$\mathregular{_3}$ Anomaly")
    plt.tight_layout()
    # plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_no2_nd_sunset.png", format='png', dpi=150)
    #plt.show()


def sage_ii_no2_sunrise_sunset(min_lat, max_lat, min_alt, max_alt):
    """
    
    :param min_lat:
    :param max_lat:
    :param min_alt:
    :param max_alt:
    :return:
    """

    sunrise, sunset = open_data.load_sage_ii_no2(start_date='1996-1-1', end_date='2005-8-31',
                                                 min_lat=min_lat, max_lat=max_lat, sun='both')
    # Find sunrise and sunset anomalies separately
    sunset = sunset.sel(altitude=slice(min_alt, max_alt))
    monthlymeans = sunset.groupby('time.month').mean('time')
    anomalies_sunset = sunset.groupby('time.month') - monthlymeans
    anomalies_sunset = anomalies_sunset.groupby('time.month') / monthlymeans

    sunrise = sunrise.sel(altitude=slice(min_alt, max_alt))
    monthlymeans = sunrise.groupby('time.month').mean('time')
    anomalies_sunrise = sunrise.groupby('time.month') - monthlymeans
    anomalies_sunrise = anomalies_sunrise.groupby('time.month') / monthlymeans

    for alt_ind in range((max_alt - min_alt) * 2):
        combined = np.array([anomalies_sunrise.NO2[:, alt_ind].values, anomalies_sunset.NO2[:, alt_ind].values])
        combined = np.nanmean(combined, axis=0)

        plt.ioff()
        sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
        sns.set_palette('Set2', 3)
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.plot(anomalies_sunrise['time'], anomalies_sunrise.NO2[:, alt_ind], 'o', markersize=5, label="Sunrise")

        plt.plot(anomalies_sunset['time'], anomalies_sunset.NO2[:, alt_ind], 'o', markersize=5, label="Sunset")

        plt.plot(anomalies_sunset['time'], combined, 'o-', markersize=5, label="Combined")

        plt.ylabel("NO$\mathregular{_2}$ Anomaly")
        plt.title("%.1f km, lat=(-5, 5)" % sunset.altitude.values[alt_ind])
        plt.legend(loc=1, frameon=True)
        plt.savefig("/home/kimberlee/Masters/NO2/Figures/SAGE_10S10N/sage_no2_%.1f.png" % sunset.altitude.values[alt_ind], format='png', dpi=150)


def sage_ii_no2_time_series(min_lat, max_lat):
    """
    :param min_lat: lower latitude limit
    :param max_lat: upper latitude limit
    :return plot monthly mean NO2 in latitude band
    """
    # load the data
    sage = SAGEIILoaderV700()
    sage.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage.load_data('1996-1-1', '2005-8-31', min_lat, max_lat)
    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    # Create datasets of monthly mean anomalies
    no2 = np.copy(data['NO2'])
    no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
    sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2)}, coords={'altitude': no2_alts, 'time': dates})
    sunset = sunset.sel(altitude=slice(24.5, 40.5))
    sunset /= 1e9
    sunset = sunset.resample(time='MS').mean('time', skipna=True)
    monthlymeans = sunset.groupby('time.month').mean('time')
    anomalies_sunset = sunset.groupby('time.month') - monthlymeans
    anomalies_sunset = anomalies_sunset.groupby('time.month') / monthlymeans

    no2_2 = data['NO2']
    no2_2[data['Type_Tan'] == 1] = np.nan  # remove sunset events
    sunrise = xr.Dataset({'NO2': (['time', 'altitude'], no2_2)}, coords={'altitude': no2_alts, 'time': dates})
    sunrise = sunrise.sel(altitude=slice(24.5, 40.5))
    sunrise /= 1e9
    sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
    monthlymeans = sunrise.groupby('time.month').mean('time')
    anomalies_sunrise = sunrise.groupby('time.month') - monthlymeans
    anomalies_sunrise = anomalies_sunrise.groupby('time.month') / monthlymeans

    combined = np.zeros((len(anomalies_sunset.time), len(anomalies_sunset.altitude)))

    for alt_ind in range(32):
        c = np.array([anomalies_sunrise.NO2[:, alt_ind].values, anomalies_sunset.NO2[:, alt_ind].values])
        combined[:, alt_ind] = np.nanmean(c, axis=0)

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    fax = plt.pcolormesh(anomalies_sunset.time, anomalies_sunset.altitude, combined.T,
                         vmin=-0.3, vmax=0.3, cmap='RdBu_r')
    plt.ylabel('Altitude [km]')
    plt.ylim(25, 40)
    plt.xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    plt.title('Monthly Mean SAGE II NO$\mathregular{_2}$ Combined (%i to %i deg. lat.)' % (min_lat, max_lat))
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    # cb.set_label("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    cb.set_label("NO$\mathregular{_2}$ Anomaly")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_no2_anomaly_combined_overlap.png", format='png', dpi=150)
    plt.show()


def sage_osiris_no2_time_series(min_lat, max_lat):
    # load sage II
    sage = SAGEIILoaderV700()
    sage.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage.load_data('1998-1-1', '2005-8-31', min_lat, max_lat)
    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    no2 = np.copy(data['NO2'])
    no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
    sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2)},
                        coords={'altitude': no2_alts,
                                'time': dates})
    sunset = sunset.sel(altitude=slice(24.5, 39.5))
    sunset /= 1e9
    sunset = sunset.resample(time='MS').mean('time', skipna=True)
    sunset = sunset.where((sunset.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS
    monthlymeans = sunset.groupby('time.month').mean('time')
    sunset_anom = sunset.groupby('time.month') - monthlymeans
    sunset_anom = sunset_anom.groupby('time.month') / sunset.groupby('time.month').std('time')

    no2_2 = np.copy(data['NO2'])
    no2_2[data['Type_Tan'] == 1] = np.nan  # remove sunset events
    sunrise = xr.Dataset({'NO2': (['time', 'altitude'], no2_2)},
                         coords={'altitude': no2_alts,
                                 'time': dates})
    sunrise = sunrise.sel(altitude=slice(24.5, 39.5))
    sunrise /= 1e9
    sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
    sunrise = sunrise.where((sunrise.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS
    monthlymeans = sunrise.groupby('time.month').mean('time')
    sunrise_anom = sunrise.groupby('time.month') - monthlymeans
    sunrise_anom = sunrise_anom.groupby('time.month') / sunrise.groupby('time.month').std('time')

    # load OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20081231'))

    no2_osiris = datafile.NO2_concentration.where((datafile.latitude > min_lat) &
                                                                     (datafile.latitude < max_lat))
    no2_osiris = no2_osiris.sel(altitude=slice(24.5, 40.5))
    no2_osiris = no2_osiris.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris *= 6.022140857e17
    no2_osiris /= 1e9
    monthlymeans = no2_osiris.groupby('time.month').mean('time')
    osiris_anom = no2_osiris.groupby('time.month') - monthlymeans
    osiris_anom = osiris_anom.groupby('time.month') / no2_osiris.groupby('time.month').std('time')

    no2_osiris_daily = datafile.derived_daily_mean_NO2_concentration.where((datafile.latitude > min_lat) &
                                                                     (datafile.latitude < max_lat))
    no2_osiris_daily = no2_osiris_daily.sel(altitude=slice(24.5, 40.5))
    no2_osiris_daily = no2_osiris_daily.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris_daily *= 6.022140857e17
    no2_osiris_daily /= 1e9
    monthlymeans = no2_osiris_daily.groupby('time.month').mean('time')
    osiris_daily_anom = no2_osiris_daily.groupby('time.month') - monthlymeans
    osiris_daily_anom = osiris_daily_anom.groupby('time.month') / no2_osiris_daily.groupby('time.month').std('time')

    no2_osiris_630 = datafile.derived_0630_NO2_concentration.where(
        (datafile.latitude > -10) & (datafile.latitude < 10))
    no2_osiris_630 = no2_osiris_630.sel(altitude=slice(24.5, 40.5))
    no2_osiris_630 = no2_osiris_630.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris_630 *= 6.022140857e17
    no2_osiris_630 /= 1e9
    monthlymeans = no2_osiris_630.groupby('time.month').mean('time')
    osiris_anom_630 = no2_osiris_630.groupby('time.month') - monthlymeans
    osiris_anom_630 = osiris_anom_630.groupby('time.month') / no2_osiris_630.groupby('time.month').std('time')

    for alt_ind in range(len(sunset.altitude)):
        plt.ioff()
        sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
        sns.set_palette('hls', 5)
        fig, ax = plt.subplots(figsize=(15, 7))
        plt.plot(sunrise_anom['time'], sunrise_anom.NO2[:, alt_ind], 'o', label="SAGE Sunrise")
        plt.plot(sunset_anom['time'], sunset_anom.NO2[:, alt_ind], 'o', label="SAGE Sunset")

        plt.plot(osiris_anom['time'], osiris_anom[:, alt_ind], '*', label="OSIRIS Daily Mean")
        plt.plot(osiris_anom_630['time'], osiris_anom_630[:, alt_ind], '*', label="OSIRIS 6:30")
        plt.plot(osiris_daily_anom['time'], osiris_daily_anom[:, alt_ind], '*', label="OSIRIS Measured")

        # plt.ylabel("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
        plt.ylabel("NO$\mathregular{_2}$ Anomaly")
        plt.title("%.1f km, lat=(%i, %i)" % (sunset.altitude.values[alt_ind], min_lat, max_lat))
        plt.legend(loc=1, frameon=True)
        #plt.savefig("/home/kimberlee/Masters/NO2/Figures/SAGE_OSIRIS_anom_timeseries/no2_65S55S_%.1f.png" %
        #            sunset.altitude.values[alt_ind], format='png', dpi=150)
        plt.show()


def sage_osiris_no2_profile(min_lat, max_lat):
    # load sage II
    sage = SAGEIILoaderV700()
    sage.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage.load_data('2002-1-1', '2005-8-31', min_lat, max_lat)
    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    no2 = np.copy(data['NO2'])
    no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
    sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2)},
                        coords={'altitude': no2_alts,
                                'time': dates})
    sunset = sunset.sel(altitude=slice(24.5, 39.5))
    sunset /= 1e9
    sunset = sunset.resample(time='MS').mean('time', skipna=True)
    sunset = sunset.where((sunset.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS

    no2_2 = np.copy(data['NO2'])
    no2_2[data['Type_Tan'] == 1] = np.nan  # remove sunset events
    sunrise = xr.Dataset({'NO2': (['time', 'altitude'], no2_2)},
                         coords={'altitude': no2_alts,
                                 'time': dates})
    sunrise = sunrise.sel(altitude=slice(24.5, 39.5))
    sunrise /= 1e9
    sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
    sunrise = sunrise.where((sunrise.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS

    # all SAGE
    sage_all = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2'])},
                          coords={'altitude': no2_alts,
                                  'time': dates})
    sage_all = sage_all.sel(altitude=slice(24.5, 39.5))
    sage_all /= 1e9
    sage_all = sage_all.resample(time='MS').mean('time', skipna=True)
    sage_all = sage_all.where((sage_all.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS

    # load OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20050831'))

    no2_osiris = datafile.NO2_concentration.where((datafile.latitude > min_lat) & (datafile.latitude < max_lat))
    no2_osiris = no2_osiris.sel(altitude=slice(24.5, 40.5))
    no2_osiris = no2_osiris.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris *= 6.022140857e17
    no2_osiris /= 1e9

    no2_osiris_daily = datafile.derived_daily_mean_NO2_concentration.where((datafile.latitude > min_lat)
                                                                           & (datafile.latitude < max_lat))
    no2_osiris_daily = no2_osiris_daily.sel(altitude=slice(24.5, 40.5))
    no2_osiris_daily = no2_osiris_daily.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris_daily *= 6.022140857e17
    no2_osiris_daily /= 1e9

    no2_osiris_630 = datafile.derived_0630_NO2_concentration.where(
        (datafile.latitude > -10) & (datafile.latitude < 10))
    no2_osiris_630 = no2_osiris_630.sel(altitude=slice(24.5, 40.5))
    no2_osiris_630 = no2_osiris_630.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris_630 *= 6.022140857e17
    no2_osiris_630 /= 1e9

    for month in range(len(sunset.time.values)):
        plt.ioff()
        sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
        sns.set_palette('hls', 5)
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.plot(sunrise.NO2[month, :], sunrise.altitude, '-o', label="SAGE Sunrise")
        plt.plot(sunset.NO2[month, :], sunset.altitude, '-o', label="SAGE Sunset")
        # plt.plot(sage_all['time'], sage_all.NO2[:, alt_ind], 'o', label="SAGE Both")

        plt.plot(no2_osiris_daily[month, :], no2_osiris_daily.altitude, '-*', label="OSIRIS Daily Mean")
        plt.plot(no2_osiris_630[month, :], no2_osiris_630.altitude, '-*', label="OSIRIS 6:30")
        plt.plot(no2_osiris[month, :], no2_osiris.altitude, '-*', label="OSIRIS Measured")

        plt.xlabel("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
        plt.ylabel('Altitude [km]')
        plt.xlim([0.2, 2.6])

        t = pd.to_datetime(str(sunset.time.values[month]))
        timestring = t.strftime('%Y-%B')
        plt.title("%s, lat=(%i, %i)" % (timestring, min_lat, max_lat))
        plt.legend(loc=1, frameon=True)
        plt.savefig("/home/kimberlee/Masters/NO2/Figures/SAGE_OSIRIS_profiles/no2_5S5N_%i_%s.png" %
                    (month, timestring), format='png', dpi=150)


def sage_osiris_no2_im(min_lat, max_lat):
    # load sage II
    sage = SAGEIILoaderV700()
    sage.data_folder = '/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/'
    data = sage.load_data('1996-1-1', '2005-8-31', min_lat, max_lat)
    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))
    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan  # uncertainty greater than 100%
    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    no2 = np.copy(data['NO2'])
    no2[data['Type_Tan'] == 0] = np.nan  # remove sunrise events
    sunset = xr.Dataset({'NO2': (['time', 'altitude'], no2)},
                        coords={'altitude': no2_alts,
                                'time': dates})
    sunset = sunset.sel(altitude=slice(24.5, 39.5))
    sunset /= 1e9
    sunset = sunset.resample(time='MS').mean('time', skipna=True)
    sunset = sunset.where((sunset.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS
    monthlymeans = sunset.groupby('time.month').mean('time')
    sunset_anom = sunset.groupby('time.month') - monthlymeans
    sunset_anom = sunset_anom.groupby('time.month') / monthlymeans

    no2_2 = np.copy(data['NO2'])
    no2_2[data['Type_Tan'] == 1] = np.nan  # remove sunset events
    sunrise = xr.Dataset({'NO2': (['time', 'altitude'], no2_2)},
                         coords={'altitude': no2_alts,
                                 'time': dates})
    sunrise = sunrise.sel(altitude=slice(24.5, 39.5))
    sunrise /= 1e9
    sunrise = sunrise.resample(time='MS').mean('time', skipna=True)
    sunrise = sunrise.where((sunrise.altitude % 1) == 0.5, drop=True)  # every second altitude to match OSIRIS
    monthlymeans = sunrise.groupby('time.month').mean('time')
    sunrise_anom = sunrise.groupby('time.month') - monthlymeans
    sunrise_anom = sunrise_anom.groupby('time.month') / monthlymeans

    # load OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20161231'))
    no2_osiris = datafile.NO2_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    no2_osiris = no2_osiris.sel(altitude=slice(24.5, 40.5))
    no2_osiris = no2_osiris.resample(time='MS').mean('time', skipna=True)
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2_osiris *= 6.022140857e17
    no2_osiris /= 1e9
    monthlymeans = no2_osiris.groupby('time.month').mean('time')
    osiris_anom = no2_osiris.groupby('time.month') - monthlymeans
    osiris_anom = osiris_anom.groupby('time.month') / monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True, figsize=(12, 10))

    plt.ioff()

    ax1.pcolormesh(sunrise_anom.time, sunrise_anom.altitude, sunrise_anom.NO2.T, vmin=-0.3, vmax=0.3, cmap='RdBu_r')
    f2 = ax2.pcolormesh(sunset_anom.time, sunset_anom.altitude, sunset_anom.NO2.T, vmin=-0.3, vmax=0.3, cmap='RdBu_r')
    ax3.pcolormesh(osiris_anom.time, osiris_anom.altitude, osiris_anom.T, vmin=-0.3, vmax=0.3, cmap='RdBu_r')

    ax1.set_ylabel('Altitude [km]')
    ax2.set_ylabel('Altitude [km]')
    ax3.set_ylabel('Altitude [km]')
    ax1.set_ylim(25, 40)
    ax2.set_ylim(25, 40)
    ax3.set_ylim(25, 40)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    ax3.set_xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    ax1.set_title('Monthly Mean SAGE II NO$\mathregular{_2}$ - Sunrise (-5 to 5 deg. lat.)')
    ax2.set_title('Monthly Mean SAGE II NO$\mathregular{_2}$ - Sunset (-5 to 5 deg. lat.)')
    ax3.set_title('Monthly Mean OSIRIS NO$\mathregular{_2}$ - Measured (-5 to 5 deg. lat.)')

    f.subplots_adjust(right=0.9)
    cbar_ax = f.add_axes([0.91, 0.11, 0.01, 0.77])
    cb = f.colorbar(f2, cax=cbar_ax)
    cb.set_label("NO$\mathregular{_2}$ Anomaly")
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_osiris_no2_anomaly.png", format='png',
                dpi=150)

    plt.show()


if __name__ == "__main__":
    sage_ii_no2_sunrise_sunset(-5, 5, 24.5, 40.5)
    # sage_ii_no2_time_series(-5, 5)
    # sage_osiris_no2_time_series(-25, 15)
    # sage_osiris_no2_profile(-5, 5)
    # sage_osiris_no2_im(-5, 5)
