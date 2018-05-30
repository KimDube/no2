###
# Plot the Singapore monthly zonal mean wind from 2005 to 2014.
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr


heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
# Acquire wind data
location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
file1 = pd.read_csv(location)
file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]

time = pd.date_range('2002-01-01', freq='M', periods=12 * 14)
# make into an xarray cause i like a challenge?
winds = xr.DataArray(file1, coords=[heights, time], dims=['heights', 'time'])
w1 = winds.sel(time=slice('2005-01-01', '2007-08-01'))  # nov 2005
w2 = winds.sel(time=slice('2007-01-01', '2009-08-01'))  # nov 2007
w3 = winds.sel(time=slice('2009-08-01', '2012-03-01'))  # jun 2010
w4 = winds.sel(time=slice('2011-10-01', '2014-05-01'))  # sep 2012

wavr = np.zeros(np.shape(w1))
for i in range(len(w1.time)):
    for j in range(len(w1.heights)):
        wavr[j, i] = np.mean([w1.values[j, i], w2.values[j, i], w3.values[j, i], w4.values[j, i]])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 8))
im = plt.contourf(np.arange(-10, 21), w1.heights, wavr*0.1, np.arange(-40, 40, 2.5), extend='both')
# winds in units of 0.1 m/s
plt.plot([0, 0], [w1.heights[0], w1.heights[-1]], 'k-')

glines = [10, 20, 30, 40, 50, 60, 70, 80, 90]
for i in glines:
    plt.plot([-10, 20], [i, i], 'w-', linewidth=0.5)

ax.set_yscale('log')
from matplotlib.ticker import ScalarFormatter
ax.yaxis.set_major_formatter(ScalarFormatter())
ax.yaxis.set_major_locator(plt.FixedLocator(heights))
plt.gca().invert_yaxis()
plt.ylabel("Pressure [hPa]")
plt.xlabel("Time Lag [Months]")
plt.title("Singapore Monthly Mean Zonal Wind")
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("Wind Speed [m/s]")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_average.png", format='png', dpi=150)
plt.show()
