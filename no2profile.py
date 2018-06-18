###
# Open and produce OSIRIS NO2 number density or vmr profiles
###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import xarray as xr


if __name__ == "__main__":
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20050101', '20141231'))

    tropics = datafile.derived_0630_NO2_concentration.where((datafile.latitude > -5) & (datafile.latitude < 5))
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    tropics *= 6.022140857e17

    # To convert number density to vmr
    # pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
    # temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))
    # tropics = (tropics * temp * 1.3806503e-19 / pres) * 1e9  # ppbv

    tropics = tropics.resample('YS', dim='time', how='mean')
    tropics /= 1e9
    avr_profile = tropics.mean(dim='time')
    std_profile = tropics.std(dim='time')

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.semilogx(avr_profile, avr_profile.altitude)
    plt.semilogx(avr_profile + std_profile, avr_profile.altitude, linewidth=0.5)
    plt.semilogx(avr_profile - std_profile, avr_profile.altitude, linewidth=0.5)
    plt.xlim([0.07, 1.05])
    plt.ylim([15, 40])
    plt.xlabel("NO$\mathregular{_2}$ [10$\mathregular{^{9}}$ molecules/cm$\mathregular{^{3}}$]")
    # plt.xlabel("NO$\mathregular{_2}$ VMR [ppbv]")
    plt.ylabel("Altitude [km]")
    plt.title("Annual Average NO$\mathregular{_2}$ @ 6:30 (-5 to 5 deg. lat.)")

    plt.grid(True, which="both", ls="-")
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/OSIRIS_avrNO2prof_2005_14.png", format='png', dpi=150)
    plt.show()

    """
    # load csv file from database
    loc = '/home/kimberlee/Masters/NO2/O3_NO2_2007.csv'
    header = ['MJD', 'LocalSolarTime', 'Altitude', 'AirDensity', 'Latitude', 'Longitude', 'O3BoundedNumberDensity',
              'NO2BoundedNumberDensity']
    datafile = pd.read_csv(loc, names=header)
    # don't care about longitude. Or local solar time?
    datafile = datafile.drop(['LocalSolarTime', 'Longitude'], axis=1)

    # set up datetime index
    datafile['Datetime'] = pd.to_datetime(datafile['MJD'], unit='d', origin=pd.Timestamp('1858-11-17'))
    datafile = datafile.set_index('Datetime')
    datafile = datafile.drop(['MJD'], axis=1)

    # remove latitudes outside range of interest
    datafile = datafile.loc[(datafile.Latitude >= -5) & (datafile.Latitude <= 5)]
    # don't need latitude anymore
    datafile = datafile.drop(['Latitude'], axis=1)

    # take monthly mean of each number density at each altitude
    alts = np.arange(15500, 41500, 1000)
    monthlymeans = pd.DataFrame()
    for i in range(len(alts)):
        f = datafile.loc[datafile.Altitude == alts[i]]
        # remove any negative number densities so they don't ruin the mean
        f = f[(f['NO2BoundedNumberDensity'] >= 0)]
        f = f.resample('M').mean()
        monthlymeans = pd.concat([monthlymeans, f])

    # To create a profile average together the monthly values at each altitude
    meanprofile = np.zeros(len(alts))
    stdprofile = np.zeros(len(alts))
    for i in range(len(alts)):
        f = monthlymeans.loc[monthlymeans.Altitude == alts[i]]
        meanprofile[i] = np.mean(f['NO2BoundedNumberDensity'])
        stdprofile[i] = np.std(f['NO2BoundedNumberDensity'])
    """
