###
# NO, NO2, NOx, HNO3, N2O VMR profiles
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import numpy as np


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


if __name__ == "__main__":
    # NO2, NO, NOx from OSIRIS
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20050101', '20141231'))

    no2 = datafile.derived_0630_NO2_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    # to convert concentration to number density [mol/m^3 to molecule/cm^3]
    no2 *= 6.022140857e17
    nox = datafile.derived_0630_NOx_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    nox *= 6.022140857e17
    no = datafile.derived_0630_NO_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
    no *= 6.022140857e17

    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
    temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))

    no2 = (no2 * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
    no2 = no2.resample('YS', dim='time', how='mean')
    avr_no2 = no2.mean(dim='time')

    nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
    nox = nox.resample('YS', dim='time', how='mean')
    avr_nox = nox.mean(dim='time')

    no = (no * temp * 1.3806503e-19 / pres) * 1e9  # ppbv
    no = no.resample('YS', dim='time', how='mean')
    avr_no = no.mean(dim='time')

    # N2O from MLS
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    n2o = datafile.N2O.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    n2o['nLevels'] = datafile.Pressure.mean('Time')
    n2o *= 1e9  # ppbv
    n2o = n2o.resample('YS', dim='Time', how='mean')
    avr_n2o = n2o.mean(dim="Time")

    # HNO3 from MLS
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    hno3 = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10))
    hno3['nLevels'] = datafile.Pressure.mean('Time')
    hno3 *= 1e9  # ppbv
    hno3 = hno3.resample('YS', dim='Time', how='mean')
    avr_hno3 = hno3.mean(dim="Time")

    # Calculate NOy = HNO3 + NOx
    nox_pres = pres.mean(dim='time')
    mls_levels = datafile.Pressure.mean('Time')
    heights_index = []
    for i in mls_levels[5:17]:
        n = find_nearest(nox_pres, i)
        heights_index.append(int(n.values))  # (nox["altitude"].values[n])

    noy = avr_hno3[5:17].values + avr_nox[heights_index].values

    # Plot
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 8))
    l1 = ax.semilogx(avr_no2, avr_no2.altitude, sns.xkcd_rgb["magenta"], marker='o', label="NO$\mathregular{_2}$")
    l2 = ax.semilogx(avr_nox, avr_nox.altitude, sns.xkcd_rgb["golden"], marker='o', label="NO$\mathregular{_x}$")
    l3 = ax.semilogx(avr_no, avr_no.altitude, sns.xkcd_rgb["medium green"], marker='o', label="NO")

    ax2 = ax.twinx()
    l4 = ax2.loglog(avr_n2o, avr_n2o.nLevels, sns.xkcd_rgb["azure"], marker='D', label="N$\mathregular{_2}$O")
    l5 = ax2.loglog(avr_hno3, avr_hno3.nLevels, sns.xkcd_rgb["bright orange"], marker='D', label="HNO$\mathregular{_3}$")
    l6 = ax2.loglog(noy, mls_levels.values[5:17], sns.xkcd_rgb["light purple"], marker='D', label="NO$\mathregular{_y}$")
    ax.set_ylim([15, 40])
    ax2.set_ylim([3, 120])
    ax2.set_ylim(ax2.get_ylim()[::-1])

    ax.set_xlabel("VMR [ppbv]")
    ax.set_ylabel("Altitude [km] - OSIRIS NO, NO2, NOx")
    ax2.set_ylabel("Pressure [hPa] - MLS N2O, HNO3, MLS+OSIRIS NOy")
    plt.title("Annual Average (-10 to 10 deg. lat.)")

    ax2.grid(True, which="both", ls="-")
    ax.grid(True, which="major", ls="-", axis="x")
    ax.grid(True, which="major", ls="--", axis="y")

    l = ax.legend(loc=2)
    l2 = ax2.legend(loc=1)

    plt.savefig("/home/kimberlee/Masters/NO2/Figures/nitrogenspecies.png", format='png', dpi=150)
    plt.show()

