
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


from pysagereader.sage_ii_reader import SAGEIILoaderV700

import xarray as xr


def sage_ii_no2_time_series(data_folder):
    """
    plot the monthly averaged no2 in the tropics
    """

    # load the data
    sage = SAGEIILoaderV700()
    sage.data_folder = data_folder
    data = sage.load_data('1984-10-24', '1994-8-31', -10, 10)

    # convert dates
    dates = pd.to_datetime(data['mjd'], unit='d', origin=pd.Timestamp('1858-11-17'))

    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 1000] = np.nan  # uncertainty greater than 10%
    data['NO2'][data['Type_Sat'] == 0] = np.nan  # remove sunrise events

    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    # Create dataset
    ds = xr.Dataset({'NO2': (['time', 'altitude'], data['NO2'])},
                    coords={'altitude': no2_alts,
                    'time': dates})

    monthly_no2 = ds.resample(time='MS').mean('time')
    monthlymeans = monthly_no2.groupby('time.month').mean('time')
    anomalies = monthly_no2.groupby('time.month') - monthlymeans
    anomalies = anomalies.groupby('time.month') / monthlymeans

    sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
    fig, ax = plt.subplots(figsize=(10, 5))
    fax = anomalies.NO2.T.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-0.5, 0.5, 0.01),
                                        extend='both', add_colorbar=0, cmap="RdBu_r")
    plt.ylabel('Altitude [km]')
    plt.ylim(20, 50)
    plt.xlabel('')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=0)
    # plt.title('Monthly Mean SAGE II O$\mathregular{_3}$ (-10 to 10 deg. lat.)')
    plt.title('Monthly Mean SAGE II NO$\mathregular{_2}$ @ Sunset (-10 to 10 deg. lat.)')
    cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
    # cb.set_label("O$\mathregular{_3}$ Anomaly")
    cb.set_label("NO$\mathregular{_2}$ Anomaly")
    plt.tight_layout()
    plt.savefig("/home/kimberlee/Masters/NO2/Figures/sage_no2_anomaly.png", format='png', dpi=150)
    plt.show()


if __name__ == "__main__":
    sage_ii_no2_time_series('/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/')