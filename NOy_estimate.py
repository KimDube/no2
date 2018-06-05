import numpy as np
import xarray as xr
import seaborn as sns
import matplotlib.pyplot as plt


def find_nearest(array, value):
    """
    Find location of array element closest to value
    :param array: array to search
    :param value: number to find
    :return: index corresponding to closest element of array to value
    """
    index = (np.abs(array-value)).argmin()
    return index


# load NOx
datafile = xr.open_mfdataset('/home/kimberlee/OsirisData/Level2/no2_v6.0.2/*.nc')
datafile = datafile.swap_dims({'profile_id': 'time'}, inplace=True)

nox = datafile.derived_0630_NOx_concentration.where((datafile.latitude > -10) & (datafile.latitude < 10))
# To convert concentration to number density [mol/m^3 to molecule/cm^3]
nox *= 6.022140857e17
# To convert number density to vmr
pres = datafile.pressure.where((datafile.latitude > -10) & (datafile.latitude < 10))
temp = datafile.temperature.where((datafile.latitude > -10) & (datafile.latitude < 10))
nox = (nox * temp * 1.3806503e-19 / pres) * 1e6  # ppmv
nox = nox.resample('MS', dim='time', how='mean')

pres = pres.mean(dim='time')
nox_avr = nox.mean(dim='time').values

# Load HNO3
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20051231'))
mls_levels = datafile.Pressure.mean('Time')
hno3 = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10)) * 1e6
hno3_avr = hno3.mean(dim='Time').values


heights_index = []
for i in mls_levels[7:15]:
    n = find_nearest(pres, i)
    heights_index.append(int(n.values))  # (nox["altitude"].values[n])

print(np.shape(hno3))

noy_avr = hno3_avr[7:15] + nox_avr[heights_index]

# time by height
noy_time = np.zeros((12, 8))
for i in range(12):
    print(i)
    noy_time[i, :] = hno3.values[i, 7:15] + nox.values[i, heights_index]


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})

# plt.imshow(noy_time, aspect='auto')

#plt.loglog(noy_avr, mls_levels.values[7:15])
#plt.loglog(hno3_avr[7:15], mls_levels.values[7:15])
#plt.loglog(nox_avr[heights_index], mls_levels.values[7:15])

#for i in mls_levels:
#    plt.semilogy([10, 60], [i, i], '-k', linewidth=0.5)

#lt.ylim([4, 70])
#plt.xlabel("VMR [ppbv]")
#plt.ylabel("pressure [hPa]")
#plt.gca().invert_yaxis()
plt.show()

