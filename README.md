# no2

Assorted files for the analysis of:
- OSIRIS NOx
- OSIRIS O3
- MLS N2O 
- MLS HNO3
- MLS O3

NO2_alt_time_plt.py
- plot NO2 monthly mean anomaly time series as a function of altitude for a specified latitude band.

O3_NOx_correlation.py
- Correlation coefficient as a function of altitude for O3 and NOx.

O3_alt_time_plt.py
- plot O3 monthly mean anomaly time series as a function of altitude for a specified latitude band.

combine_hno3.py
- Load a set of HNO3 files (created with read_mls_HNO3.py) and take monthly mean. Save new file.

hno3_alt_time_plt.py
- plot HNO3 monthly mean anomaly time series as a function of altitude for a specified latitude band.

n2o_alt_time_plt.py
- plot N2O monthly mean anomaly time series as a function of altitude for a specified latitude band.

nitrogen_partitioning.py
- plot average profiles of mixing ratios for various nitrogen containing compounds. 

no2profile.py
- plot an average NO2 profile in units of either number density or VMR.

nox_alt_time_plt.py
- plot NOx monthly mean anomaly time series as a function of altitude for a specified latitude band.

qbo_contours.py
- plot the Singapore monthly zonal mean wind from 2005 to 2014.

qbo_lag.py
- plot average Singapore winds as a function of time since QBO onset.

qbo_lag_NOx.py
- plot average N2O anomaly as a function of time since QBO onset. Overlplot Singapore winds. 

qbo_lag_O3.py
- plot average ozone anomaly as a function of time since QBO onset. Overlplot Singapore winds. 

read_mls_HNO3.py
- open a set of MLS HNO3. Remove un-needed parameters. Save new, smaller, files with values in a specified latitude band. Follow use with combine_hno3.py to get monthly means.

read_mls_N2O.py
- open a set of MLS N2O files and calculate the monthly mean N2O VMR in a specified latitude band for each available pressure level. Save new files.

test.py
- practice code. Probably doesn't work.

timeseries_10hPa.py
- plot anomalies over time at 10 hPa (~32 km) of OSIRS O3, MLS O3, OSIRIS NOx, MLS HNO3, MLS N2O
