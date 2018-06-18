
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from NO2 import helper_functions, open_data

# Load NOx
nox, pres_nox = open_data.load_osiris_nox_monthly(start_date='20050101', end_date='20141231',
                                                  min_lat=-10, max_lat=10, pressure=1)

# get values at 10 hPa
nox_10_hpa = np.zeros(len(nox.time))
for i in range(len(nox.time)):
    pressure_i = pres_nox[i, :]
    nox_i = nox[i, :]
    n, l = helper_functions.interpolate_to_mls_pressure(pressure_i, nox_i)
    # l = 10 hPa at index 7, l = 21.5 hPa at index 5
    nox_10_hpa[i] = n[7]

nox_10_hpa_dataset = xr.DataArray(nox_10_hpa, coords=[nox.time], dims=["time"])
monthlymeans = nox_10_hpa_dataset.groupby('time.month').mean('time')
anomalies_nox = nox_10_hpa_dataset.groupby('time.month') - monthlymeans

# Load O3
o3, pres_o3 = open_data.load_osiris_ozone_monthly(start_date='20050101', end_date='20141231',
                                                  min_lat=-10, max_lat=10, pressure=1)

# get values at 10 hPa
o3_10_hpa = np.zeros(len(o3.time))
for i in range(len(o3.time)):
    pressure_i = pres_o3[i, :]
    o3_i = o3[i, :]
    n2, l2 = helper_functions.interpolate_to_mls_pressure(pressure_i, o3_i)
    # l = 10 hPa at index 7, l = 21.5 hPa at index 5
    o3_10_hpa[i] = n2[7]

o3_10_hpa_dataset = xr.DataArray(o3_10_hpa, coords=[o3.time], dims=["time"])
monthlymeans = o3_10_hpa_dataset.groupby('time.month').mean('time')
anomalies_o3 = o3_10_hpa_dataset.groupby('time.month') - monthlymeans


sns.set(context="talk", style="white", rc={'font.family': [u'serif']})
fig, ax = plt.subplots(figsize=(8, 7))
sns.regplot(y=anomalies_o3, x=anomalies_nox, color=sns.xkcd_rgb["bright orange"], ci=None,
            marker='D', scatter_kws={"s": 50})

ax.set_xlabel("OSIRIS NO$\mathregular{_x}$ (ppbv)")
ax.set_ylabel("OSIRIS O$\mathregular{_3}$ (ppmv)")
plt.title("10.0 hPa")
ax.set_xlim([-2.2, 2.2])
ax.set_ylim([-1.2, 1.2])

plt.tight_layout()
plt.savefig("/home/kimberlee/Masters/NO2/Figures/NOx_v_O3_10hPa.png", format='png', dpi=150)
plt.show()