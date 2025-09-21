from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
import glob
import shap
import os

project_dir = os.getcwd()  # or replace 'os.getcwd()' with the specific path if needed

# Define the path to the 'irkutsk.csv' file
irkutsk_filepath = os.path.join(project_dir, 'irkutsk.csv')


train_data = pd.read_csv(irkutsk_filepath)
print(train_data.head(), "\n", train_data.size)
train_data['time'] = train_data['time'].apply(lambda x: int(x.split(':')[0]))

train_data['date'] = pd.to_datetime(train_data['date'], format='%m/%d/%Y', errors='coerce')
train_data['year'] = train_data['date'].dt.year
train_data['month'] = train_data['date'].dt.month
train_data['day'] = train_data['date'].dt.day
train_data['day_of_week'] = train_data['date'].dt.dayofweek

columns_to_clean = ['plan_value_hps', 'plan_value_tps', 'tech_min_hps', 'tech_min_tps',
    'technology_min_hps', 'technology_min_tps', 'tech_max_hps',
    'tech_max_tps', 'plan_value', 'plan_exp', 'plan_imp',
    'price_demand', 'price_supply', 'ov_plan_value',
    'infl', 'price_demand_real', 'price_supply_real',
    'real_usd_rub', 'oil_rub_real', 'coal_rub_real',
    'al_rub_real', 'steel_rub_real', 'year', 'month', 'day', 'day_of_week']

# Remove commas and convert to float
for column in columns_to_clean:
    train_data[column] = train_data[column].astype(str)
    train_data[column] = train_data[column].str.replace(',', '').astype(float)
    train_data[column] = pd.to_numeric(train_data[column], errors='coerce')



X = train_data.drop(['region', 'price_demand_real', 'price_supply_real', 'price_demand', 'price_supply', 'date'], axis=1)
y = train_data['price_demand_real']

#print(y.value_counts())
#print(X.dtypes)  # Viewing the data types of features in X
#print(X.isnull().sum())  # Check for any null values in X

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Обучим модель на полных тренировочных данных
param_grid = {
    'max_depth': [5, 10, 15, 20, 25, 30],
    'min_samples_leaf': [1, 2, 5, 10, 15, 20],
    'max_iter': [50, 100, 125, 150, 200, 250],
    'learning_rate': [0.01, 0.05, 0.1, 0.2]
}

# Create the model
model = HistGradientBoostingRegressor(random_state=42)

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=2, n_jobs=-1)

grid_search.fit(X, y)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = abs(grid_search.best_score_)  # Take the absolute MSE

print("Best Parameters:", best_params)
print("Best Cross-Validated MSE Score:", best_score)

# Optional: Fit the best model on full data for future predictions
best_model = grid_search.best_estimator_
best_model.fit(X, y)
