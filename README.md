# Hybrid Weather Forecasting Framework

A two-stage hybrid framework for generating realistic, probabilistic precipitation forecasts by combining deterministic initial forecasting with advanced U-Net spatial models.

In the first stage, a deterministic model generates an initial baseline forecast using historical meteorological data. In the second stage, this baseline is concatenated with the original observed features and passed to a generative diffusion model, which functions as a probabilistic refiner by rectifying the deterministic error and providing the parameters of a Gaussian distribution (mean (`μ`) and log-variance (`log(σ²)`)) for every grid point. The U-Net model functions as the probabilistic refiner and the performance has been compared for `4` different variants of the same—the standard U-Net, residual U-Net, attention U-Net and the CBAM (Convolutional Block Attention Module) U-Net. In general, the hybrid approach significantly improved the performance of all tested deterministic models, with lower RMSE, higher ACC, and the generation of reliable probabilistic forecasts measured by CRPS.

Link for PyTorch Installation: https://pytorch.org/get-started/locally/

## Repository Structure

<details>
<summary><b>Click to expand the repository structure</b></summary>

<br>

```text
hybrid-weather-forecaster/
├── utils/
│   ├── Base_Test.py                           # Basic Testing Modules
│   ├── Baseline_Model_Definitions.py          # Baseline Model Definitions
│   ├── Baseline_Model_Utils.py                # Baseline Model Training Modules
│   ├── Feature_Target_Scaler_RH.ipynb         # Scaling Features and Target - Relative Humidity
│   ├── Feature_Target_Scaler_TP.ipynb         # Scaling Features and Target - Total Precipitation
│   ├── Feature_Target_Scaler_Temp.ipynb       # Scaling Features and Target - Temperature
│   ├── KC_Data_Preprocessing.py               # Data Preprocessing Module
│   ├── Test_Plot_Codes.py                     # Test Result Plotting Modules
│   ├── UNet_Model_Definitions.py              # U-Net Model Definitions
│   └── UNet_Model_Utils.py                    # U-Net Model Training Module
├── training/
│   ├── ElasticNet_U-Net_Hybrid_RH_Training_KC_4S.ipynb      # Elastic Net - U-Net Hybrid Training (RH - 4 Step Input)
│   ├── ElasticNet_U-Net_Hybrid_RH_Training_KC_8S.ipynb      # Elastic Net - U-Net Hybrid Training (RH - 8 Step Input)
│   ├── ElasticNet_U-Net_Hybrid_TP_Training_KC_4S.ipynb      # Elastic Net - U-Net Hybrid Training (TP - 4 Step Input)
│   ├── ElasticNet_U-Net_Hybrid_TP_Training_KC_8S.ipynb      # Elastic Net - U-Net Hybrid Training (TP - 8 Step Input)
│   ├── ElasticNet_U-Net_Hybrid_Temp_Training_KC_4S.ipynb    # Elastic Net - U-Net Hybrid Training (Temp - 4 Step Input)
│   ├── ElasticNet_U-Net_Hybrid_Temp_Training_KC_8S.ipynb    # Elastic Net - U-Net Hybrid Training (Temp - 8 Step Input)
│   ├── GRU_U-Net_Hybrid_RH_Training_KC_4S.ipynb             # GRU - U-Net Hybrid Training (RH - 4 Step Input)
│   ├── GRU_U-Net_Hybrid_RH_Training_KC_8S.ipynb             # GRU - U-Net Hybrid Training (RH - 8 Step Input)
│   ├── GRU_U-Net_Hybrid_TP_Training_KC_4S.ipynb             # GRU - U-Net Hybrid Training (TP - 4 Step Input)
│   ├── GRU_U-Net_Hybrid_TP_Training_KC_8S.ipynb             # GRU - U-Net Hybrid Training (TP - 8 Step Input)
│   ├── GRU_U-Net_Hybrid_Temp_Training_KC_4S.ipynb           # GRU - U-Net Hybrid Training (Temp - 4 Step Input)
│   ├── GRU_U-Net_Hybrid_Temp_Training_KC_8S.ipynb           # GRU - U-Net Hybrid Training (Temp - 8 Step Input)
│   ├── LSTM_U-Net_Hybrid_RH_Training_KC_4S.ipynb            # LSTM - U-Net Hybrid Training (RH - 4 Step Input)
│   ├── LSTM_U-Net_Hybrid_RH_Training_KC_8S.ipynb            # LSTM - U-Net Hybrid Training (RH - 8 Step Input)
│   ├── LSTM_U-Net_Hybrid_TP_Training_KC_4S.ipynb            # LSTM - U-Net Hybrid Training (TP - 4 Step Input)
│   ├── LSTM_U-Net_Hybrid_TP_Training_KC_8S.ipynb            # LSTM - U-Net Hybrid Training (TP - 8 Step Input)
│   ├── LSTM_U-Net_Hybrid_Temp_Training_KC_4S.ipynb          # LSTM - U-Net Hybrid Training (Temp - 4 Step Input)
│   ├── LSTM_U-Net_Hybrid_Temp_Training_KC_8S.ipynb          # LSTM - U-Net Hybrid Training (Temp - 8 Step Input)
│   ├── NN_U-Net_Hybrid_RH_Training_KC_4S.ipynb              # Neural Network - U-Net Hybrid Training (RH - 4 Step Input)
│   ├── NN_U-Net_Hybrid_RH_Training_KC_8S.ipynb              # Neural Network - U-Net Hybrid Training (RH - 8 Step Input)
│   ├── NN_U-Net_Hybrid_TP_Training_KC_4S.ipynb              # Neural Network - U-Net Hybrid Training (TP - 4 Step Input)
│   ├── NN_U-Net_Hybrid_TP_Training_KC_8S.ipynb              # Neural Network - U-Net Hybrid Training (TP - 8 Step Input)
│   ├── NN_U-Net_Hybrid_Temp_Training_KC_4S.ipynb            # Neural Network - U-Net Hybrid Training (Temp - 4 Step Input)
│   └── NN_U-Net_Hybrid_Temp_Training_KC_8S.ipynb            # Neural Network - U-Net Hybrid Training (Temp - 8 Step Input)
├── EDA.ipynb                                  # Exploratory Data Analysis
├── README.md                                  # Project Description
└── requirements.txt                           # Python dependencies
```

</details>

## Dataset Description

The dataset used for this study has been sourced from ERA5, which is the fifth generation ECMWF reanalysis for the global climate and weather for the past `8` decades, with the data being updated daily with a latency of `5` days. It provides hourly estimates for a large number of atmospheric, ocean-wave and land-surface quantities.

Link for this dataset: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview

| Attribute | Details |
| :---: | :---: |
| **Data Type** | Gridded |
| **Projection** | Regular latitude-longitude grid |
| **Horizontal Coverage** | Global |
| **Horizontal Resolution** | `0.25° x 0.25°` (atmosphere) <br> `0.5° x 0.5°` (ocean waves) |
| **Temporal Coverage** | `1940` – present |
| **Temporal Resolution** | Hourly |
| **File Format** | GRIB |
| **Update Frequency** | Daily |

For the purpose of this study, a subset of the above dataset has been used, which contains weather data for the coastal regions of Maharashtra, Goa, and Karnataka for the years `2001` to `2025`.

Link: https://drive.google.com/file/d/1gjqpWTHyzhBaH1TKQiwoYFczUVhxwDqX/view?usp=sharing

| Attribute | Description |
| :---: | :---: |
| **Location** | Latitude: `12.00°N` – `19.75°N` <br> Longitude: `72.75°E` – `75.50°E` |
| **Number of Samples** | `14,025,216` |
| **Sampling Rate** | `6` hours |
| **Number of Attributes** | `8` |
| **Input Features** | Weather parameters such as temperature, relative humidity, surface pressure, etc. |
| **Target Variable** | Total precipitation (`tp`) (or) Temperature (`t2m`) (or) Relative humidity (`rh`) |
| **Data Type(s)** | `float64` |

| Attribute | Mean | Standard Deviation | Minimum | Median | Maximum |
| :---: | :---: | :---: | :---: | :---: | :---: |
| *msl* | `100,989.091` | `301.673` | `98,038.896` | `100,991.188` | `102,195.385` |
| *sst* | `27.746` | `1.351` | `22.876` | `27.905` | `32.127` |
| *u10* | `1.669` | `2.929` | `-19.905` | `1.625` | `22.973` |
| *v10* | `-0.549` | `2.169` | `-19.517` | `-0.369` | `22.299` |
| *ws* | `3.556` | `2.054` | `0.064` | `3.113` | `24.657` |
| *t2m* | `26.360` | `3.449` | `6.730` | `26.700` | `42.858` |
| *rh* | `72.294` | `18.243` | `6.652` | `77.055` | `100.006` |
| *tp* | `1.257` | `3.580` | `0.000` | `0.003` | `228.680` |

## Performance Metrics

To compare the performance of these models, root mean squared error (RMSE), anomaly correlation coefficient (ACC) and continuous ranked probability score (CRPS) have been used as the performance metrics. The performance is evaluated before refining (using the deterministic model), as well as after refining (using the probabilistic U-Net). For the deterministic models, the forecast is a single point estimate rather than a distribution. In this special case, CRPS simplifies to the mean absolute error (MAE).

## Model Configurations

<table>
  <thead>
    <tr>
      <th align="center">Category</th>
      <th align="center">Model</th>
      <th align="center">Architecture / Core Layers</th>
      <th align="center">Total Parameters</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4" align="center"><b>Baseline Models</b></td>
      <td align="center"><b>Elastic Net</b></td>
      <td align="center">Linear</td>
      <td align="center"><code>4,718,976</code></td>
    </tr>
    <tr>
      <td align="center"><b>Neural Network</b></td>
      <td align="center">Linear → ReLU → Linear</td>
      <td align="center"><code>405,920</code></td>
    </tr>
    <tr>
      <td align="center"><b>LSTM</b></td>
      <td align="center">LSTM → Linear</td>
      <td align="center"><code>204,288</code></td>
    </tr>
    <tr>
      <td align="center"><b>GRU</b></td>
      <td align="center">GRU → Linear</td>
      <td align="center"><code>154,848</code></td>
    </tr>
    <tr>
      <td rowspan="4" align="center"><b>U-Net Models</b></td>
      <td align="center"><b>Standard U-Net</b></td>
      <td align="center">—</td>
      <td align="center"><code>7,703,682</code></td>
    </tr>
    <tr>
      <td align="center"><b>Residual U-Net</b></td>
      <td align="center">—</td>
      <td align="center"><code>8,051,138</code></td>
    </tr>
    <tr>
      <td align="center"><b>Attention U-Net</b></td>
      <td align="center">—</td>
      <td align="center"><code>7,791,275</code></td>
    </tr>
    <tr>
      <td align="center"><b>CBAM U-Net</b></td>
      <td align="center">—</td>
      <td align="center"><code>7,747,594</code></td>
    </tr>
  </tbody>
</table>

<br>

| Hyperparameter | Baseline Models | U-Net Models |
| :---: | :---: | :---: |
| **Optimizer** | AdamW | AdamW |
| **Learning Rate** | `1e-3` | `1e-4` |
| **Weight Decay** | `0` | `0.01` |
| **Scheduler** | ReduceLROnPlateau | ReduceLROnPlateau |
| **Factor** | `0.5` | `0.1` |
| **Patience (Epochs)** | `5` | `3` |
