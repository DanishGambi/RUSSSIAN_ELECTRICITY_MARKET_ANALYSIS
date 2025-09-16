# RUSSSIAN_ELECTRICITY_MARKET_ANALYSIS


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

📈 Model Performance
The regression models achieved high accuracy across all regions, as measured by R² score:

Region	Mean Squared Error (MSE)	R² Score
Moscow Oblast	~1622.71	0.983
Republic of Tatarstan	~1328.86	0.979
Chelyabinsk Oblast	~1153.79	0.974
Irkutsk Oblast	~1207.37	0.966

🛠️ Technology Stack
Programming Language: Python
Key Libraries: shap (for analysis), scikit-learn (for model building), pandas, numpy, matplotlib
Visualization Dashboard: Yandex DataLens
