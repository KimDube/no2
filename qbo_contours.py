###
# Plot the Singapore monthly zonal mean wind from 2005 to 2014.
###

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import seaborn as sns


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
# Convert pressure levels to altitude (rough values)
t = np.loadtxt("/home/kimberlee/Masters/Data_Other/alt_pres.txt", delimiter=' ')
alts = t[:, 0]
pres = t[:, 1]

heights_m = []
for i in heights:
    n = find_nearest(pres, i)
    heights_m.append(alts[n])

heights_m = np.array(heights_m)
heights_m /= 1000

# Acquire wind data
location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
file1 = pd.read_csv(location)
file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]
file = np.zeros(np.shape(file1))

startdate = datetime.date(2002, 1, 1)
days = []
while startdate.year < 2016:
    if startdate.day == 1:
        days.append(startdate)
    startdate += datetime.timedelta(days=1)

# Find January of each year to help with marking qbo phase
jan = []
for i in range(len(days)):
    if days[i].month == 1:
        jan.append(i)

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(10, 5))
# im = plt.contourf(days, heights_m, file1*0.1, np.arange(-40, 40, 2.5), extend='both', cmap='BuPu')
im2 = plt.contour(days, heights, file1*0.1, 8, linewidths=1)  # winds in units of 0.1 m/s

plt.ylim([10, 100])
plt.xlim([days[36], days[-12]])

plt.plot([days[0], days[-1]], [heights[3], heights[3]], 'k-.')
ax.set_yscale('log')

# Plot lines at onset of westerly (positive) winds at 20 hPa
plt.plot([days[jan[3] + 10], days[jan[3] + 10]], [heights[0], heights[-1]], 'k-')  # Nov 2005
plt.plot([days[jan[5] + 10], days[jan[5] + 10]], [heights[0], heights[-1]], 'k-')  # Nov 2007
plt.plot([days[jan[8] + 5], days[jan[8] + 5]], [heights[0], heights[-1]], 'k-')  # Jun 2010
plt.plot([days[jan[10] + 8], days[jan[10] + 8]], [heights[0], heights[-1]], 'k-')  # Sep 2012
plt.grid(True, which="both", ls="-")
plt.clabel(im2, inline=1, fontsize=8)
plt.gca().invert_yaxis()
plt.ylabel("Pressure [hPa]")
plt.title("Singapore Monthly Mean Zonal Wind")
# cb = fig.colorbar(im2, orientation='horizontal', fraction=0.2, aspect=50)
plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO.png", format='png', dpi=150)
plt.show()
