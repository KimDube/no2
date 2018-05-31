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

pres = pres.mean(dim='time')
nox = nox.mean(dim='time').values

# Load HNO3
datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_HNO3_monthlymeans/MLS-HNO3-*.nc')
datafile = datafile.sel(Time=slice('20050101', '20051231'))
mls_levels = datafile.Pressure.mean('Time')
hno3 = datafile.HNO3.where((datafile.Latitude > -10) & (datafile.Latitude < 10)) * 1e6
hno3 = hno3.mean(dim='Time').values


heights_index = []
for i in mls_levels[7:15]:
    n = find_nearest(pres, i)
    heights_index.append(int(n.values))  # (nox["altitude"].values[n])


print(hno3[7:15])
print(nox[heights_index])

noy = hno3[7:15] + nox[heights_index]
print(noy)

sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
# plt.semilogy(nox["altitude"], pres, '-o')

plt.loglog(noy, mls_levels.values[7:15])
plt.loglog(hno3[7:15], mls_levels.values[7:15])
plt.loglog(nox[heights_index], mls_levels.values[7:15])

#for i in mls_levels:
#    plt.semilogy([10, 60], [i, i], '-k', linewidth=0.5)

plt.ylim([4, 70])
plt.xlabel("VMR [ppbv]")
plt.ylabel("pressure [hPa]")
plt.gca().invert_yaxis()
plt.show()

