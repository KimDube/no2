import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time

from pysagereader.sage_ii_reader import SAGEIILoaderV700


def sage_ii_no2_time_series(data_folder):
    """
    Basic example of using the loader to plot the monthly averaged no2 in the tropics
    """

    # load the data
    sage = SAGEIILoaderV700()
    sage.data_folder = data_folder
    data = sage.load_data('2000-1-1', '2005-8-31', -20, 20)

    # setup the time bins
    time_res = 30
    mjds = np.arange(np.min(data['mjd']), np.max(data['mjd']), time_res)

    # get rid of bad data
    data['NO2'][data['NO2'] == data['FillVal']] = np.nan
    data['NO2'][data['NO2_Err'] > 10000] = np.nan

    # get no2 altitudes
    no2_alts = data['Alt_Grid'][(data['Alt_Grid'] >= data['Range_NO2'][0]) & (data['Alt_Grid'] <= data['Range_NO2'][1])]

    # average the no2 profiles
    no2 = np.zeros((len(no2_alts), len(mjds)))
    for idx, mjd in enumerate(mjds):
        good = (data['mjd'] > mjd) & (data['mjd'] < mjd + time_res)
        with warnings.catch_warnings():  # there will be all nans at high/low altitudes, its fine
            warnings.simplefilter("ignore", category=RuntimeWarning)
            no2[:, idx] = np.nanmean(data['NO2'][good, :], axis=0)

    plot_data(no2_alts, mjds + time_res/2, no2)


def plot_data(alts, mjds, val):

    # make the plot
    plt.contourf(Time(mjds, format='mjd').datetime, alts, val, levels=np.arange(0, 4e9, 1e8), extend='both')
    plt.colorbar()
    #plt.clim(0,7e12)
    plt.ylabel('Altitude [km]')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylim(15, 40)


if __name__ == "__main__":
    sage_ii_no2_time_series('/home/kimberlee/ValhallaData/SAGE/Sage2_v7.00/')
    plt.show()