# 650_Report
This project implements gap-filling of MODIS Land Surface Temperature (LST) using low-rank tensor decomposition. The provided dataset MODIS_Aug.mat is a 100×200×31 tensor (August 2020), where missing values (clouds) are encoded as zeros


Two scripts are included:

Tucker_Decomposition.m

  Tucker + EM refinement, and optional bootstrap ensemble for uncertainty.
  
  The result shown is Day 31, selected because the large missing region makes the comparison clearer. The day index can be changed inside the script.
  
CP_Decomposition.m

  CP baseline under the same settings.
  
  ！！！Note: CP requires hours to run (very slow), while Tucker completes in 1–2 minutes.

  
Users can modify rank, EM iterations, M, and p_keep to evaluate sensitivity and obtain better metrics (RMSE, R², CRPS).
