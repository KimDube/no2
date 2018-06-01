###
# Plot the average NOx anomaly as lag from QBO onset
###

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import pandas as pd


# load wind data
heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
heights_m = np.array([31, 30, 29, 27, 25, 24, 23, 22, 21, 21, 20, 19, 18, 17, 17])
# Acquire wind data
location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
file1 = pd.read_csv(location)
file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]

time = pd.date_range('2002-01-01', freq='M', periods=12 * 14)
# make into an xarray cause i like a challenge?
winds = xr.DataArray(file1, coords=[heights, time], dims=['heights', 'time'])
w1 = winds.sel(time=slice('2005-01-01', '2007-09-01'))  # nov 2005
w2 = winds.sel(time=slice('2007-01-01', '2009-09-01'))  # nov 2007
w3 = winds.sel(time=slice('2009-08-01', '2012-04-01'))  # jun 2010
w4 = winds.sel(time=slice('2011-10-01', '2014-06-01'))  # sep 2012

wavr = np.zeros(np.shape(w1))
for i in range(len(w1.time)):
    for j in range(len(w1.heights)):
        wavr[j, i] = np.mean([w1.values[j, i], w2.values[j, i], w3.values[j, i], w4.values[j, i]])


# load and filter NOx data
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

oz1 = anomalies.sel(time=slice('2005-01-01', '2007-08-01')).values  # nov 2005
oz2 = anomalies.sel(time=slice('2007-01-01', '2009-08-01')).values  # nov 2007
oz3 = anomalies.sel(time=slice('2009-08-01', '2012-03-01')).values  # jun 2010
oz4 = anomalies.sel(time=slice('2011-10-01', '2014-05-01')).values  # sep 2012

wavr_oz = np.zeros(np.shape(oz1))
for i in range(32):
    for j in range(30):
        wavr_oz[i, j] = np.nanmean([oz1[i, j], oz2[i, j], oz3[i, j], oz4[i, j]])

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 8))
im = plt.contourf(np.arange(-10, 22), anomalies.altitude, wavr_oz.transpose(), np.arange(-0.8, 0.81, 0.05),
                  extend='both', cmap='RdBu_r')
im2 = plt.contour(np.arange(-10, 22), heights_m, (wavr*0.1), 8, extend='both', colors='black')
plt.plot([0, 0], [anomalies.altitude[0], anomalies.altitude[-1]], 'k-')
ax.text(3, 28, 'W', fontsize=24)
ax.text(-6, 27.5, 'E', fontsize=24)
plt.ylim([15, 40])
plt.grid(True, which="both", ls="-")
plt.ylabel("Altitude [km]")
plt.xlabel("Time Lag from Onset of Westerly QBO @ 20 hPa [Months]")
plt.title("OSIRIS NO$\mathregular{_x}$")
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("NO$\mathregular{_x}$ VMR Anomaly [ppbv]")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_average_NOx.png", format='png', dpi=150)
plt.show()
