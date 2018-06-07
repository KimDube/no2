###
# Plot O3 time series as function of altitude with singapore wind contours
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from NO2 import open_data


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

# Load O3
o3 = open_data.load_osiris_ozone_monthly(start_date='20050101', end_date='20141231', min_lat=-5, max_lat=5)
monthlymeans = o3.groupby('time.month').mean('time')
anomalies = o3.groupby('time.month') - monthlymeans

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(10, 5))
fax = anomalies.plot.contourf(x='time', y='altitude', robust=True, levels=np.arange(-1.2, 1.2, 0.1),
                              cmap="RdBu_r", extend='both', add_colorbar=0)
im2 = winds.plot.contour(x='time', y='heights', levels=8, extend='both', cmap='gray')
plt.clabel(im2, inline=1, fontsize=8)
plt.ylim([15, 40])
plt.ylabel('Altitude [km]')
plt.xlabel('')
plt.title('Monthly Mean OSIRIS O$\mathregular{_3}$ (-5 to 5 deg. lat.)')
sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
cb = fig.colorbar(fax, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("O$\mathregular{_3}$ Anomaly VMR [ppmv]")
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/NO2/Figures/altVtime_O3.png", format='png', dpi=150)
plt.show()

