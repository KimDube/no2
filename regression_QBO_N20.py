###
# Fit QBO wind components to O3
###


import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from statsmodels.formula.api import ols
from NO2 import open_data


heights = np.array([10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
# Acquire wind data
location = '/home/kimberlee/Masters/Data_Other/singapore.csv'
file1 = pd.read_csv(location)
file1 = np.array(file1)  # rows are heights and columns are month/year [height, time]
time = pd.date_range('2002-01-01', freq='M', periods=12 * 14)
winds = xr.DataArray(file1, coords=[heights, time], dims=['heights', 'time'])
winds = 0.1 * winds.sel(time=slice('2005-01-01', '2014-12-31')).T  # winds in units of 0.1 m/s

pca = PCA(n_components=2)
wind_transf = pca.fit_transform(winds)

# Standardize for regression
std_qbo1 = np.nanstd(wind_transf[:, 0])
std_qbo2 = np.nanstd(wind_transf[:, 1])
qbo1 = (wind_transf[:, 0] - np.nanmean(wind_transf[:, 0])) / std_qbo1
qbo2 = (wind_transf[:, 1] - np.nanmean(wind_transf[:, 1])) / std_qbo2

# Create a set of alt by time matrices for 10 degree latitude bins
lats = np.arange(-60, 70, 10)
mls_levels = np.load('/home/kimberlee/Masters/NO2/mls_pressure_levels.npy')
alts = mls_levels[5:15]

reg_coeff_qbo1 = np.zeros((len(lats), len(alts)))
reg_coeff_qbo2 = np.zeros((len(lats), len(alts)))
p_val_qbo1 = np.zeros((len(lats), len(alts)))
p_val_qbo2 = np.zeros((len(lats), len(alts)))

for i in range(len(lats)-1):
    print('lat bin =  %i' % i)
    # Load monthly mean NOx in lat range.
    datafile = xr.open_mfdataset('/home/kimberlee/Masters/NO2/MLS_N2O_monthlymeans/MLS-N2O-*.nc')
    datafile = datafile.sel(Time=slice('20050101', '20141231'))
    # ***** monthly mean files are filtered to only contain data in tropics. Need to make new versions for this to work
    n2o = datafile.N2O.where((datafile.Latitude > lats[i]) & (datafile.Latitude < lats[i+1]))
    n2o['nLevels'] = datafile.Pressure.mean('Time')
    n2o *= 1e9  # ppbv
    n2o = n2o.where(n2o['nLevels'] <= 147, drop=True)

    for j in range(len(alts)):
        print('alt bin = %i' % j)

        print(n2o[:-3, j].values)
        exit()

        # apply regression to nox[:, j]
        n2o_1 = (n2o[:-3, j] - np.nanmean(n2o[:-3, j])) / np.nanstd(n2o[:-3, j])  # standardize

        y = pd.Series(n2o_1)
        n2o_1 = y.interpolate(method='linear')  # fill in missing values
        df = pd.DataFrame(data={'N2O': n2o_1, 'QBO1': qbo1, 'QBO2': qbo2})
        df = df.dropna()

        model = ols(formula='N2O ~ QBO1 + QBO2', data=df).fit()
        p_val_qbo1[i, j] = model.pvalues[1]
        p_val_qbo2[i, j] = model.pvalues[2]
        reg_coeff_qbo1[i, j] = model.params[1]
        reg_coeff_qbo2[i, j] = model.params[2]


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 8))
im = plt.contourf(lats, alts, reg_coeff_qbo1.T, np.arange(-0.7, 0.75, 0.05), extend='both', cmap='RdYlBu_r')
# mark insignificant regions
for i in range(len(lats)-1):
    for j in range(len(alts)-2):
        if p_val_qbo1[i, j] > 0.05:
            plt.text(lats[i]+5, alts[j], 'X', fontsize=10)

ax.set_ylim([15, 39])
plt.ylabel("Altitude [km]")
plt.xlabel("Latitude")
plt.title('MLS N2O - QBO1')
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("Regression Coeff.")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_1_N2O.png", format='png', dpi=150)
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 8))
im = plt.contourf(lats, alts, reg_coeff_qbo2.T, np.arange(-0.7, 0.76, 0.05), extend='both', cmap='RdYlBu_r')
# mark insignificant regions
for i in range(len(lats)-1):
    for j in range(len(alts)-2):
        if p_val_qbo2[i, j] > 0.05:
            plt.text(lats[i]+5, alts[j], 'X', fontsize=10)

ax1.set_ylim([15, 39])
plt.ylabel("Altitude [km]")
plt.xlabel("Latitude")
plt.title('MLS N2O - QBO2')
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("Regression Coeff.")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_2_N2O.png", format='png', dpi=150)
plt.show()