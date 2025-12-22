# Analysis of Electricity Consumption Across Russian Regions

**Authors:** Dmitrii A. Maliuzhantsev, Arina M. Tarasova, Anna K. Andaralova, Polina S. Belokorovii, Dariana A. Salchak  
**Scientific Supervisor:** Aleksandr Yu. Filatov  

**Affiliation:** Department of Socio-Economic Research and Regional Development, School of Economics and Management, Far Eastern Federal University (FEFU)

## 📜 Overview
This repository contains code and analysis for a research project focused on forecasting and understanding the factors driving electricity consumption in four key Russian regions:

* **Moscow Oblast**
* **Irkutsk Oblast**
* **Chelyabinsk Oblast**
* **Republic of Tatarstan**

The project employs machine learning regression models to predict consumption and uses **SHAP (SHapley Additive exPlanations) analysis** to interpret the model predictions and identify the most impactful features. An interactive dashboard built with Yandex DataLens is also provided for visual exploratory data analysis.

## 🌟 Primary Results: SHAP Analysis Visualizations

The **core analytical output** of this project is a comprehensive set of **42 SHAP plots** across four regions, providing deep insights into feature importance and model interpretability. These visualizations are stored in the `visuals/` directory and represent the key findings of our research.

### 📁 Visualization Structure
```
visuals/
├── Moscow/
│ ├── shap_summary_plot.png
│ ├── shap_summary_bar_plot.png
│ ├── dependence_plot_feature_0.png
│ ├── dependence_plot_feature_1.png
│ └── ... (+38 dependence plots)
├── Irkutsk/
│ ├── shap_summary_plot.png
│ ├── shap_summary_bar_plot.png
│ ├── dependence_plot_feature_0.png
│ ├── dependence_plot_feature_1.png
│ └── ... (+38 dependence plots)
├── Chelyabinsk/
│ └── ... (same structure)
└── Tatarstan/
└── ... (same structure)
```


### 🔍 SHAP Plot Types

For **each region**, we provide three types of SHAP visualizations:

1. **Summary Plots (`summary_plot_<region>.png`)**
   - Shows feature importance rankings
   - Displays the distribution of SHAP values for each feature
   - Colors represent feature values (red=high, blue=low)

2. **Bar Plots (`bar_plot_<region>.png`)**
   - Bar chart of mean absolute SHAP values
   - Clear ranking of most influential features
   - Quantitative measure of feature impact

3. **Dependence Plots (`dependence_plot_<feature>_<region>.png`)**
   - **40 plots per region** showing individual feature effects
   - Reveals how predictions change with feature values
   - Shows interactions between features
   - Key features analyzed include:
     - Time features (hour, day_of_week, month)
     - Weather variables (temperature, wind speed)
     - Economic indicators (oil_price, coal_price)
     - Production data from various power plants
     - Historical consumption patterns

### 📈 Key Insights from SHAP Analysis

**Common Patterns Across Regions:**
- **Time of day** (hour) consistently emerges as the most important feature
- **Seasonal patterns** (month) show strong cyclical behavior
- **Historical consumption** features demonstrate high importance for predictions

**Region-Specific Findings:**
- **Moscow Oblast:** Strong influence of economic activity patterns and population density
- **Irkutsk Oblast:** Industrial factors and resource prices play significant roles
- **Chelyabinsk Oblast:** Manufacturing indicators show higher importance
- **Republic of Tatarstan:** Mixed industrial-agricultural patterns visible in feature importance

## ❓ Research Question
What are the main factors influencing electricity consumption in different regions of the Russian Federation?

## 📊 Key Findings / Abstract
Our analysis reveals that **time of day and season (month) are the most significant factors** influencing electricity demand, confirming strong diurnal and seasonal patterns. Key insights include:

* **Winter peaks** are driven by heating needs, particularly evident in northern regions
* **Summer increases** are linked to air conditioning usage, especially in urban areas
* **Planned production data** from various power plants are crucial for understanding consumption dynamics
* **Industrial factors** like coal and steel prices have an indirect but noticeable impact, especially in resource-rich regions like Irkutsk Oblast
* **Moscow Oblast** leads in both production and consumption due to high population density and economic activity, despite a lack of local natural resources

The SHAP analysis provides **transparent, quantitative evidence** for these findings, showing exactly how much each factor contributes to consumption predictions.

## 🎯 Model Performance

The regression models achieved high accuracy across all regions, as measured by R² score:

| Region | Mean Squared Error (MSE) | R² Score |
| :--- | :--- | :--- |
| **Moscow Oblast** | ~1622.71 | **0.983** |
| **Republic of Tatarstan** | ~1328.86 | **0.979** |
| **Chelyabinsk Oblast** | ~1153.79 | **0.974** |
| **Irkutsk Oblast** | ~1207.37 | **0.966** |

**Note:** The high R² scores indicate excellent predictive performance, while the SHAP analysis ensures these models are interpretable and their predictions are explainable.

## 🛠️ Technology Stack

*   **Programming Languages:** Python, R
*   **Key Libraries:** `shap` (for analysis), `scikit-learn` (for model building), `pandas`, `numpy`, `matplotlib`, `seaborn`
*   **Visualization Dashboard:** Yandex DataLens
*   **Model Interpretability:** SHAP (SHapley Additive exPlanations)

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
```
### Usage

1.  **Prepare Data:** Place your regional CSV files (`moscow.csv`, `irkutsk.csv`, etc.) in a `data/` directory. The data should include features like timestamp, planned production types, weather data, and economic indicators.

2.  **Generate SHAP Plots:** Execute the main analysis script:
    ```bash
    python scripts/shap_analysis.py
    ```
    The script will:
    *   Train regression models for each region
    *   Calculate SHAP values for model interpretability
    *   Generate all 42 visualization plots in the `visuals/` directory
    *   Save regional comparison summaries

3.  **Explore Results:** Navigate to the `visuals/` directory to view:
    *   Regional summary plots for quick overview
    *   Detailed dependence plots for specific feature analysis
    *   Comparative analysis across regions

### Note on Model Tuning

The model hyperparameters were optimized using a search grid on the Irkutsk Oblast dataset and then applied to all other regions to maintain consistency. The SHAP analysis validates that the models are learning meaningful patterns rather than overfitting.

## 📊 Interactive Dashboard

The Yandex DataLens dashboard provides interactive visualizations for deeper analysis. It consists of multiple tabs:

1.  **Main Tab:** Scatter plots (2020-2021, 2022, 2023-2024) showing the relationship between price and planned volume for all four regions.
2.  **Comparative Tabs:** Side-by-side graphs for comparing two selected regions, analyzing:
    *   Demand vs. Date (daily averages, holiday/seasonal peaks)
    *   Demand vs. Time of Day (diurnal patterns, morning/evening peaks)
    *   Demand vs. Day of the Week (weekday/weekend differences)
    *   Impact of Weather Conditions (temperature, wind)
    *   Resource Prices (correlation between electricity price and oil/coal prices)

*Access to the live dashboard is currently managed by the authors.*

## 🔑 Keywords

**English:** Energy demand forecasting, electricity market, electricity prices, SHAP analysis in Python, machine learning in energy, regression models for Russian regions, seasonality in energy consumption, impact of weather on energy consumption, industrial electricity consumption, model interpretability, feature importance analysis.

**Russian:** прогнозирование потребления электроэнергии, рынок электроэнергии, цены на электроэнергию, SHAP-анализ в Python, машинное обучение в энергетике, регрессионные модели для регионов России, сезонность в энергопотреблении, влияние погоды на энергопотребление, промышленное потребление электроэнергии, интерпретируемость моделей, анализ важности признаков.

## 📄 Citation

If you use this code, visualizations, or findings in your research, please cite:

> Maliuzhantsev D.A., Tarasova A.M., Andaralova A.K., Belokorovii P.S., Salchak D.A. (2024). Analysis of Electricity Consumption Across Russian Regions. School of Economics and Management, Far Eastern Federal University.  
> *Includes comprehensive SHAP analysis with 42 visualizations across four Russian regions.*

## 👥 Authors & Contact

*   **Anna K. Andaralova** - andaralova.ak@dvfu.ru
*   **Polina S. Belokorovii** - belokorovii.ps@dvfu.ru
*   **Dmitrii A. Maliuzhantsev** - malyuzhantcev.da@dvfu.ru
*   **Dariana A. Salchak** - salchak.da@dvfu.ru
*   **Arina M. Tarasova** - tarasova.am@dvfu.ru
*   **Scientific Supervisor: Aleksandr Yu. Filatov** - filatov.aiu@dvfu.ru

## ⚠️ Disclaimer

The regional datasets (`*.csv` files) are not included in this repository due to potential licensing and privacy restrictions. However, all **analysis code and visualization generation scripts** are provided, along with **42 pre-generated SHAP plots** that constitute the primary research output.
