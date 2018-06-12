# no2

Assorted files for the analysis of:
- OSIRIS NOx
- OSIRIS O3
- MLS N2O 
- MLS HNO3
- MLS O3 (in the future)
- SAGE II NO2 (in the future)


alt_time_plt_HNO3.py
- plot HNO3 monthly mean anomaly time series as a function of altitude for a specified latitude band.

alt_time_plt_N2O.py
- plot N2O monthly mean anomaly time series as a function of altitude for a specified latitude band.

alt_time_plt_NO2.py
- plot NO2 monthly mean anomaly time series as a function of altitude for a specified latitude band.

alt_time_plt_NOx.py
- plot NOx monthly mean anomaly time series as a function of altitude for a specified latitude band.

alt_time_plt_O3_osiris.py
- plot O3 monthly mean anomaly time series as a function of altitude for a specified latitude band.

correlation_N2O_NOy.py
- Calculate correlation coefficient as a function of altitude for N2O and NOy.

correlation_O3_NOy.py
- Calculate correlation coefficient as a function of altitude for O3 and NOy.

helper_functions.py
- interpolate_to_mls_pressure: Interpolate OSIRIS data from altitude to MLS pressure levels 5 to 15.
- linearinterp: fill missing values in a 2d array with linear interpolation.
- find_nearest: Find location of array element closest to an input value.

nitrogen_partitioning.py
- plot average profiles of mixing ratios for OSIRIS NO, OSIRIS NO2, OSIRIS NOx, MLS HNO3, MLS N2O, NOy=NOx+HNO3.

no2profile.py
- plot an average NO2 profile in units of either number density or VMR.

NOx_vs_O3.py
- Scatter plot of OSIRIS NOx vs O3 with best fit line.

open_data.py
- load_osiris_nox_monthly: Load OSIRIS NOx for a date and latitude range. Returns monthly mean values.
- load_osiris_ozone_monthly: Load OSIRIS O3 for a date and latitude range. Returns monthly mean values.
- read_mls_hno3: open a set of MLS HNO3. Remove un-needed parameters. Save new, smaller, files with values in a specified latitude band. Follow use with combine_mls.py to get monthly means.
- combine_mls: Load a set of filtered MLS files (from read_mls_hno3) and combine into monthly mean values. Save new file.
- read_mls_n2o: open a set of MLS N2O files and calculate the monthly mean N2O VMR in a specified latitude band for each available pressure level. Save new files.
- read_mls_o3: ** in progress (computer problems with regards to data loading)

qbo_contours.py
- plot the Singapore monthly zonal mean wind from 2005 to 2014.

qbo_lag.py
- plot average Singapore winds as a function of time since QBO onset.

qbo_lag_HNO3.py
- plot average HNO3 anomaly as a function of time since QBO onset. Over-plot Singapore winds.

qbo_lag_N2O.py
- plot average N2O anomaly as a function of time since QBO onset. Over-plot Singapore winds.

qbo_lag_NOx.py
- plot average NOx anomaly as a function of time since QBO onset. Over-plot Singapore winds.

qbo_lag_O3.py
- plot average O3 anomaly as a function of time since QBO onset. Over-plot Singapore winds.

regression_HNO3_N2O.py
- plot regression coefficient as a function of altitude fitting HNO3 to N2O.

regression_NOx_N2O.py
- plot regression coefficient as a function of altitude fitting NOx to N2O.

regression_QBO_N2O.py ** in progress (computer problems)

regression_QBO_NOx.py
- Plot regression coefficient as a function of altitude and latitude for first two principal components of zonal winds fit to O3.

regression_QBO_O3.py
- Plot regression coefficient as a function of altitude and latitude for first two principal components of zonal winds fit to O3.

scatterplot.py
- Plots of HNO3, NOx, O3, all vs. N2O for a specified pressure level.

timeseries_10hPa.py
- plot anomalies over time at 10 hPa (~32 km) of OSIRS O3, MLS O3, OSIRIS NOx, MLS HNO3, MLS N2O. Can also easily use to find time series at other pressure levels.
