###
# N2O time series
###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd


if __name__ == "__main__":
    # load wind data
    heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
    # Acquire wind data
    location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
    file1 = pd.read_csv(location)
    file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]

    time = pd.date_range('2002-01-01', freq='M', periods=12 * 14)
    winds = xr.DataArray(file1, coords=[heights, time], dims=['heights', 'time'])
    winds = 0.1 * winds.sel(time=slice('2005-01-01', '2014-12-31'))  # winds in units of 0.1 m/s

    # Load HNO3
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_O3_monthlymeans/MLS-O3-*.nc', concat_dim='nTimes', decode_times=False)
    datafile = datafile.sel(Time=slice('20050101', '20141231'))

    nox = datafile.HNO3.where((datafile.Latitude > -5) & (datafile.Latitude < 5))

    monthlymeans = nox.groupby('Time.month').mean('Time')
    anomalies = nox.groupby('Time.month') - monthlymeans

    anomalies['nLevels'] = datafile.Pressure.mean('Time')
    anomalies *= 1e6  # ppmv

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    fax = anomalies.plot.contourf(x='Time', y='nLevels', robust=True, levels=np.arange(-0.7, 0.7, 0.02),
                                  cmap="RdBu_r", extend='both', add_colorbar=0)
    im2 = winds.plot.contour(x='time', y='heights', levels=10, extend='both', cmap='gray')
    ax.set_yscale('log')
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_locator(plt.FixedLocator([5, 7, 10, 15, 20, 30, 40, 60, 80, 100]))
    plt.clabel(im2, inline=1, fontsize=8)
    plt.ylim([5, 100])
    plt.ylabel('Pressure [hPa]')
    plt.xlabel('')
    plt.title('Monthly Mean MLS O$\mathregular{_3}$ (-5 to 5 deg. lat.)')
    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    cb.set_label("O$\mathregular{_3}$ Anomaly VMR [ppbv]")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/altVtime_O3_MLS.png", format='png', dpi=150)
    plt.show()

