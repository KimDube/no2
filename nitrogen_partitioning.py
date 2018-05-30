###
# NO, NO2, NOx VMR profiles
###

import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


if __name__ == "__main__":
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

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    colours = ['magenta', 'tangerine', 'medium green']
    sns.set_palette(sns.xkcd_palette(colours))
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.semilogx(avr_no2, avr_no2.altitude, label="NO$\mathregular{_2}$")
    plt.semilogx(avr_nox, avr_nox.altitude, label="NO$\mathregular{_x}$")
    plt.semilogx(avr_no, avr_no.altitude, label="NO")
    plt.xlim([0.01, 11])
    plt.ylim([15, 40])
    plt.xlabel("VMR [ppbv]")
    plt.ylabel("Altitude [km]")
    plt.title("Annual Average @ 6:30 (-10 to 10 deg. lat.)")
    plt.legend(loc=2)

    plt.grid(True, which="both", ls="-")
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/OSIRIS_nitrogenspecies.png", format='png', dpi=150)
    plt.show()

