# Analysis of Electricity Consumption Across Russian Regions


Authors: Dmitrii A. Maliuzhantsev, Arina M. Tarasova, Anna K. Andaralova, Polina S. Belokorovii, Dariana A. Salchak
Scientific Supervisor: Aleksandr Yu. Filatov

Affiliation: Department of Socio-Economic Research and Regional Development, School of Economics and Management, Far Eastern Federal University (FEFU)

## 📜 Overview
This repository contains code and analysis for a research project focused on forecasting and understanding the factors driving electricity consumption in four key Russian regions:

* Moscow Oblast
* Irkutsk Oblast
* Chelyabinsk Oblast
* Republic of Tatarstan

The project employs machine learning regression models to predict consumption and uses SHAP (SHapley Additive exPlanations) analysis to interpret the model predictions and identify the most impactful features. An interactive dashboard built with Yandex DataLens is also provided for visual exploratory data analysis.

## ❓ Research Question
What are the main factors influencing electricity consumption in different regions of the Russian Federation?

🚀 Key Features
* Regional Models: Individual machine learning models trained and evaluated for each of the four Russian regions.
* Model Interpretability: Extensive use of SHAP analysis to explain model predictions and quantify feature importance.
* Interactive Dashboard: A multi-tab Yandex DataLens dashboard for visualizing relationships between electricity demand, time, weather, and economic factors.
* Comparative Analysis: Insights into how factors driving consumption differ between industrialized, agricultural, and high-population regions.

📊 Key Findings / Abstract
Our analysis reveals that time of day and season (month) are the most significant factors influencing electricity demand, confirming strong diurnal and seasonal patterns. Key insights include:
* Winter peaks are driven by heating needs.
* Summer increases are linked to air conditioning usage.
* Planned production data from various power plants are crucial for understanding consumption dynamics.
* Industrial factors like coal and steel prices have an indirect but noticeable impact, especially in resource-rich regions like Irkutsk Oblast.
* Moscow Oblast leads in both production and consumption due to high population density and economic activity, despite a lack of local natural resources.

## 📈 Model Performance

The regression models achieved high accuracy across all regions, as measured by R² score:

| Region | Mean Squared Error (MSE) | R² Score |
| :--- | :--- | :--- |
| **Moscow Oblast** | ~1622.71 | **0.983** |
| **Republic of Tatarstan** | ~1328.86 | **0.979** |
| **Chelyabinsk Oblast** | ~1153.79 | **0.974** |
| **Irkutsk Oblast** | ~1207.37 | **0.966** |

## 🛠️ Technology Stack

*   **Programming Language:** Python
*   **Key Libraries:** `shap` (for analysis), `scikit-learn` (for model building), `pandas`, `numpy`, `matplotlib`
*   **Visualization Dashboard:** Yandex DataLens

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.7+ installed. Install the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib shap
```

### Usage

1.  **Prepare Data:** Place your regional CSV files (`moscow.csv`, `irkutsk.csv`, etc.) in a `data/` directory. The data should include features like timestamp, planned production types, weather data, and economic indicators.
2.  **Run Analysis:** Execute the main script. It will prompt you to choose a region to analyze.
    ```bash
    python scripts/shap_analysis.py
    ```
3.  **Output:** The script will:
    *   Train a regression model on the selected data.
    *   Print performance metrics (MSE, R²).
    *   Generate and save three types of SHAP plots in the `visuals/` folder:
        *   `summary_plot_<region>.png`: Overview of feature importance.
        *   `bar_plot_<region>.png`: Bar chart of mean |SHAP values|.
        *   `dependence_plot_<feature>_<region>.png`: Shows the effect of a single feature on predictions.

### Note on Model Tuning

The model hyperparameters were optimized using a search grid on the Irkutsk Oblast dataset and then applied to all other regions to prevent potential overfitting. Future work could involve region-specific tuning.

## 📊 Dashboard

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

**English:** Energy demand forecasting, electricity market, electricity prices, SHAP analysis in Python, machine learning in energy, regression models for Russian regions, seasonality in energy consumption, impact of weather on energy consumption, industrial electricity consumption.

**Russian:** прогнозирование потребления электроэнергии, рынок электроэнергии, цены на электроэнергию, SHAP-анализ в Python, машинное обучение в энергетике, регрессионные модели для регионов России, сезонность в энергопотреблении, влияние погоды на энергопотребление, промышленное потребление электроэнергии.

## 📄 Citation

If you use this code or findings in your research, please cite the authors:
> Maliuzhantsev D.A., Tarasova A.M., Andaralova A.K., Belokorovii P.S., Salchak D.A. (2024). Analysis of Electricity Consumption Across Russian Regions. School of Economics and Management, Far Eastern Federal University.

## 👥 Authors & Contact

*   **Anna K. Andaralova** - andaralova.ak@dvfu.ru
*   **Polina S. Belokorovii** - belokorovii.ps@dvfu.ru
*   **Dmitrii A. Maliuzhantsev** - malyuzhantcev.da@dvfu.ru
*   **Dariana A. Salchak** - salchak.da@dvfu.ru
*   **Arina M. Tarasova** - tarasova.am@dvfu.ru
*   **Scientific Supervisor: Aleksandr Yu. Filatov** - filatov.aiu@dvfu.ru

## 📚 References

1.  AO "ATS" Official Website. [https://www.atsenergo.ru/](https://www.atsenergo.ru/) (Accessed: 18.11.2024)

## ⚠️ Disclaimer

The regional datasets (`*.csv` files) are not included in this repository due to potential licensing and privacy restrictions. 

