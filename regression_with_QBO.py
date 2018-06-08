###
# Fit QBO wind components to NOx
###


import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
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
qbo1 = (wind_transf[:, 0] - np.nanmean(wind_transf[:, 0])) / np.nanstd(wind_transf[:, 0])
qbo2 = (wind_transf[:, 1] - np.nanmean(wind_transf[:, 1])) / np.nanstd(wind_transf[:, 1])

# Create a set of alt by time matrices for 10(?) degree latitude bins
lats = np.arange(-60, 70, 10)
alts = np.arange(10.5, 40.5, 1)
reg_coeff_qbo1 = np.zeros((len(lats), len(alts)))
reg_coeff_qbo2 = np.zeros((len(lats), len(alts)))

for i in range(len(lats)-1):
    print('lat bin =  %i' % i)
    # Load monthly mean NOx in lat range, nox has dimensions [120, 30]
    nox = open_data.load_osiris_nox_monthly(start_date='20050101', end_date='20141231',
                                            min_lat=lats[i], max_lat=lats[i+1])
    for j in range(len(alts)):
        print('alt bin = %i' % j)
        # apply regression to nox[:, j]
        nox1 = (nox[:, j] - np.nanmean(nox[:, j])) / np.nanstd(nox[:, j])  # standardize
        y = pd.Series(nox1)
        nox1 = y.interpolate(method='linear')  # fill in missing values
        df = pd.DataFrame(data={'NOx': nox1, 'QBO1': qbo1, 'QBO2': qbo2})
        df = df.dropna()

        lm = LinearRegression()
        lm.fit(df.drop('NOx', axis=1), df['NOx'])
        reg_coeff_qbo1[i, j] = lm.coef_[0]
        reg_coeff_qbo2[i, j] = lm.coef_[1]


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 8))
im = plt.contourf(lats, alts, reg_coeff_qbo1.T, np.arange(-0.7, 0.76, 0.05), extend='both', cmap='RdBu_r')
# plt.xlim([0, 160])
plt.ylabel("Altitude [km]")
plt.xlabel("Latitude")
plt.title('OSIRIS NOx - QBO1')
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("Regression Coeff.")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_1_NOx.png", format='png', dpi=150)
plt.show()

fig, ax1 = plt.subplots(figsize=(8, 8))
im = plt.contourf(lats, alts, reg_coeff_qbo2.T, np.arange(-0.7, 0.76, 0.05), extend='both', cmap='RdBu_r')
# plt.xlim([0, 160])
plt.ylabel("Altitude [km]")
plt.xlabel("Latitude")
plt.title('OSIRIS NOx - QBO2')
cb = fig.colorbar(im, orientation='horizontal', fraction=0.2, aspect=50)
cb.set_label("Regression Coeff.")
plt.tight_layout(rect=[0, -0.1, 1, 1])
plt.savefig("/home/kimberlee/Masters/NO2/Figures/QBO_2_NOx.png", format='png', dpi=150)
plt.show()