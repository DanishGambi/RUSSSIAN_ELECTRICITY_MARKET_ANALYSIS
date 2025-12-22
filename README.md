# Analysis of Electricity Consumption Across Russian Regions

## 📁 Project Structure

```
project_root/
├── data/                           # Data directory
│   ├── moscow.csv                  # Moscow Oblast dataset
│   ├── irkutsk.csv                 # Irkutsk Oblast dataset
│   ├── chelyabinsk.csv             # Chelyabinsk Oblast dataset
│   └── tatarstan.csv               # Republic of Tatarstan dataset
├── Shap_figures/                   # Generated visualizations
│   ├── Moscow/                     # Moscow plots
│   │   ├── shap_summary_plot.png
│   │   ├── shap_summary_bar_plot.png
│   │   └── shap_dependence_plot_feature_*.png (40 files)
│   ├── Irkutsk/                    # Irkutsk plots
│   ├── Chelyabinsk/                # Chelyabinsk plots
│   └── Tatarstan/                  # Tatarstan plots
├── Irkutsk_model_parameter_tuning.py   # Hyperparameter optimization script
├── Shap_data_creation.py           # Main ML model & SHAP calculation
├── shap_statistics_visualization_instrument.py  # Visualization generator
└── README.md                       # This file
```

## 🚀 Quick Start Guide

### Prerequisites
Ensure you have Python 3.7+ installed. Install the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap xgboost
```

### Step-by-Step Usage

#### Step 1: Prepare Your Data
1. Create a `data/` directory in the project root
2. Place your regional CSV files in the `data/` directory:
   - `moscow.csv`
   - `irkutsk.csv`
   - `chelyabinsk.csv`
   - `tatarstan.csv`

**Note:** Ensure each CSV includes: timestamp, planned production types, weather data, and economic indicators.

#### Step 2: (Optional) Tune Hyperparameters
To optimize the model for a specific region (following the Irkutsk methodology):
```bash
python Irkutsk_model_parameter_tuning.py
```
**Important:** This script is configured for `irkutsk.csv`. To use with other regions, modify the script to load the corresponding CSV file.

#### Step 3: Train Model & Calculate SHAP Values
Run the main analysis script:
```bash
python Shap_data_creation.py
```
When prompted:
1. Enter the number corresponding to your target region:
   - `1` for Moscow
   - `2` for Irkutsk
   - `3` for Chelyabinsk
   - `4` for Tatarstan

The script will:
- Train a Gradient Boosting regression model
- Calculate SHAP values for interpretability
- Save results to `.npy` files (crucial for visualization)
- **Do not skip the save prompt!**

#### Step 4: Generate SHAP Visualizations
Once data is saved, run the visualization script:
```bash
python shap_statistics_visualization_instrument.py
```

This generates:
- `shap_summary_plot.png` - Feature importance with value distributions
- `shap_summary_bar_plot.png` - Mean absolute SHAP values
- 40 dependence plots (`shap_dependence_plot_feature_<index>.png`)

#### Step 5: Customize Visualizations (Optional)
Edit the last few lines of `shap_statistics_visualization_instrument.py`:

```python
# To skip summary or bar plots:
# Summary_plot(shap_values, X)  # Comment out to skip
# Bar_plot(shap_values, X)      # Comment out to skip

# To generate specific dependence plots:
Dependence_plot(shap_values, X, 0)  # Feature index 0-39
```

## 📊 Model Performance

| Region | Mean Squared Error (MSE) | R² Score |
|--------|--------------------------|----------|
| **Moscow Oblast** | ~1622.71 | **0.983** |
| **Republic of Tatarstan** | ~1328.86 | **0.979** |
| **Chelyabinsk Oblast** | ~1153.79 | **0.974** |
| **Irkutsk Oblast** | ~1207.37 | **0.966** |

## 🔍 SHAP Analysis Output

For **each region**, you'll get:

### 1. Summary Plots (`shap_summary_plot.png`)
- Feature importance ranking
- Distribution of SHAP values per feature
- Color-coded feature values (red=high, blue=low)

### 2. Bar Plots (`shap_summary_bar_plot.png`)
- Clear ranking of most influential features
- Quantitative impact measurement

### 3. Dependence Plots (40 plots)
Individual plots showing:
- How predictions change with feature values
- Feature interactions
- Analysis of:
  - Time features (hour, day_of_week, month)
  - Weather variables (temperature, wind speed)
  - Economic indicators (oil_price, coal_price)
  - Production data from power plants
  - Historical consumption patterns

## 📈 Key Insights

**Common Patterns:**
- **Time of day** (hour) is consistently the most important feature
- **Seasonal patterns** (month) show strong cyclical behavior
- **Historical consumption** features are highly predictive

**Region-Specific Findings:**
- **Moscow:** Economic activity and population density dominate
- **Irkutsk:** Industrial factors and resource prices are significant
- **Chelyabinsk:** Manufacturing indicators show higher importance
- **Tatarstan:** Mixed industrial-agricultural patterns are visible

## 🛠️ Technical Details

**Model:** Gradient Boosting Regression with optimized hyperparameters  
**Interpretability:** SHAP (SHapley Additive exPlanations)  
**Features:** 40+ including temporal, weather, economic, and production data  
**Visualization:** 168 plots total (42 per region × 4 regions)

## ⚠️ Important Notes

1. **Dataset Requirement:** Regional CSV files must follow the expected format with all required features
2. **Processing Order:** Always run `Shap_data_creation.py` before the visualization script
3. **Output Location:** All plots are saved in `Shap_figures/<region_name>/`
4. **Hyperparameter Tuning:** The provided tuning script is specifically for Irkutsk; adapt for other regions

## 👥 Authors

**Research Team:**  
Dmitrii A. Maliuzhantsev, Arina M. Tarasova, Anna K. Andaralova,  
Polina S. Belokorovii, Dariana A. Salchak  

**Supervisor:** Aleksandr Yu. Filatov  
**Affiliation:** Department of Socio-Economic Research and Regional Development,  
School of Economics and Management, Far Eastern Federal University (FEFU)

## 📄 License

Apache License 2.0 - See individual source files for specific copyright notices.

## 🔗 Contact

For questions about the code or analysis:  
malyuzhantcev.da@dvfu.ru  
danish_gambi@yahoo.com
