###
# NO2 - Number density over altitude and time
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


if __name__ == "__main__":
    # load wind data
    heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
    heights_m = np.array([31, 30, 29, 27, 25, 24, 23, 22, 21, 21, 20, 19, 18, 17, 17])
    # Acquire wind data
    location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
    file1 = pd.read_csv(location)
    file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]

    time = pd.date_range('2002-01-01', freq='M', periods=12 * 14)
    winds = xr.DataArray(file1, coords=[heights_m, time], dims=['heights', 'time'])
    winds = 0.1 * winds.sel(time=slice('2005-01-01', '2014-12-31'))  # winds in units of 0.1 m/s

    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20020101', '20161231'))
    tropics = datafile.derived_daily_mean_NO2_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    tropics = tropics.resample('MS', dim='time', how='mean')
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    tropics *= 6.022140857e17
    tropics /= 1e9

    monthlymeans = tropics.groupby('time.month').mean('time')
    anomalies = tropics.groupby('time.month') - monthlymeans
    anomalies = anomalies.groupby('time.month') / monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    #fax = anomalies.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-0.3, 0.32, 0.02),
    #                              cmap="RdBu_r", extend='both', add_colorbar=0)

    fax = plt.pcolormesh(anomalies.time, anomalies.altitude, anomalies.T, vmin=-0.3, vmax=0.3, cmap="RdBu_r")

    # im2 = winds.plot.contour(x='time', y='heights', levels=8, extend='both', cmap='gray')
    # plt.clabel(im2, inline=1, fontsize=8)
    plt.ylim([25, 40])
    plt.ylabel('Altitude [km]')
    plt.xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    plt.title('Monthly Mean OSIRIS NO$\mathregular{_2}$ (-10 to 10 deg. lat.)')
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    # cb.set_label("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    cb.set_label("NO$\mathregular{_2}$ Anomaly")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/OSIRIS_no2_anomaly_all.png", format='png', dpi=150)
    plt.show()


    """
    # load csv file from database
    # air density is divided by 1e15 so it doesn't load as a string
    loc = "/home/kimberlee/Masters/NO2/O3_NO2_2005_2014.csv"
    header = ['MJD', 'LocalSolarTime', 'Altitude', 'AirDensity', 'Latitude', 'Longitude', 'O3BoundedNumberDensity',
              'NO2BoundedNumberDensity']
    datafile = pd.read_csv(loc, names=header)
    # don't care about longitude. Or local solar time?
    datafile = datafile.drop(['LocalSolarTime', 'Longitude'], axis=1)

    # set up datetime index and get num months in data
    datafile['Datetime'] = pd.to_datetime(datafile['MJD'], unit='d', origin=pd.Timestamp('1858-11-17'))
    r = relativedelta.relativedelta(datafile['Datetime'].iloc[-1], datafile['Datetime'].iloc[0])
    nummonths = 12 * r.years + r.months + 1
    datafile = datafile.set_index('Datetime')
    datafile = datafile.drop(['MJD'], axis=1)

    # remove latitudes outside range of interest
    datafile = datafile.loc[(datafile.Latitude >= -5) & (datafile.Latitude <= 5)]
    # don't need latitude anymore
    datafile = datafile.drop(['Latitude'], axis=1)

    # take monthly mean of each number density at each altitude
    alts = np.arange(15500, 41500, 1000)
    monthlymeans = np.zeros((len(alts), nummonths))
    for i in range(len(alts)):
        f = datafile.loc[datafile.Altitude == alts[i]]
        # remove any negative number densities so they don't ruin the mean
        f = f[(f['NO2BoundedNumberDensity'] >= 0)]
        f = f.resample('M').mean()
        monthlymeans[i, :] = f['NO2BoundedNumberDensity']
        # take mean of each month of year and use to find anomaly
        for j in range(12):
            vals = monthlymeans[i, j::12]
            avr = np.mean(vals)
            monthlymeans[i, j::12] = (monthlymeans[i, j::12] - avr) / avr

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    fax = ax.contourf(np.arange(nummonths), alts / 1000, monthlymeans, np.arange(-0.4, 0.4, 0.01),
                      cmap="RdBu_r", extend='both')
    plt.ylabel('Altitude [km]')
    plt.title('Monthly Mean OSIRIS NO2 (-5 to 5 deg. lat.)')
    cb = plt.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    cb.set_label("NO2 [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    plt.tight_layout()
    plt.show()
    """

