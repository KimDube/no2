###
# NOx time series
###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd


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

    # Load NOx
    datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
    datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)
    datafile = datafile.sel(time=slice('20050101', '20141231'))

    nox = datafile.derived_0630_NOx_concentration.where((datafile.latitude > -5) & (datafile.latitude < 5))
    # To convert concentration to number density [mol/m^3 to molecule/cm^3]
    nox *= 6.022140857e17
    # To convert number density to vmr
    pres = datafile.pressure.where((datafile.latitude > -5) & (datafile.latitude < 5))
    temp = datafile.temperature.where((datafile.latitude > -5) & (datafile.latitude < 5))
    nox = (nox * temp * 1.3806503e-19 / pres) * 1e9  # ppbv

    nox = nox.resample('MS', dim='time', how='mean')
    monthlymeans = nox.groupby('time.month').mean('time')
    anomalies = nox.groupby('time.month') - monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    fax = anomalies.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-3, 3, 0.2),
                                  cmap="RdBu_r", extend='both', add_colorbar=0)
    im2 = winds.plot.contour(x='time', y='heights', levels=8, extend='both', cmap='gray')
    plt.clabel(im2, inline=1, fontsize=8)
    plt.ylim([15, 40])
    plt.ylabel('Altitude [km]')
    plt.xlabel('')
    plt.title('Monthly Mean OSIRIS NO$\mathregular{_x}$ @ 6:30 (-5 to 5 deg. lat.)')
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    cb.set_label("NO$\mathregular{_x}$ Anomaly VMR [ppbv]")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/altVtime_NOx.png", format='png', dpi=150)
    plt.show()
