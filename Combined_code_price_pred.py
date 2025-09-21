import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import shap
import os


def shap_statistics_on_model(train_data, file_name):
    print ("---------------------------------------------------------\n", file_name)
    #print(train_data.head(), "\n", train_data.size)
    train_data['time'] = train_data['time'].astype(str)
    train_data['time'] = train_data['time'].apply(lambda x: int(x.split(':')[0]))
    train_data['time'] = train_data['time'].astype(int)

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

    for column in columns_to_clean:
        train_data[column] = train_data[column].astype(str)
        train_data[column] = train_data[column].str.replace(',', '').astype(float)
        train_data[column] = pd.to_numeric(train_data[column], errors='coerce')

    X = train_data.drop(['region', 'price_demand_real', 'price_supply_real', 'price_demand', 'price_supply', 'date'],
                        axis=1)
    y = train_data['price_demand_real']

    # print(y.value_counts())
    # print(X.dtypes)  # Viewing the data types of features in X
    # print(X.isnull().sum())  # Check for any null values in X

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # Обучим модель на полных данных
    gbdt_reg = HistGradientBoostingRegressor(min_samples_leaf=10,
                                             max_depth=15,
                                             max_iter=250,
                                             learning_rate=0.1,
                                             random_state=42).fit(X, y)
    # по результатам перебора Best Parameters: {'learning_rate': 0.1, 'max_depth': 15, 'max_iter': 250, 'min_samples_leaf': 10}; Best Cross-Validated MSE Score: 25612.58657188177; param_grid = {'max_depth': [5, 10, 15, 20, 25, 30],'min_samples_leaf': [1, 2, 5, 10, 15, 20],'max_iter': [50, 100, 125, 150, 200, 250],'learning_rate': [0.01, 0.05, 0.1, 0.2]}

    y_pred = gbdt_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')

    # Using the SHAP statistics
    explainer = shap.Explainer(gbdt_reg)
    shap_values = explainer(X)
    print(f"SHAP Values Shape: {shap_values.shape}")
    return [shap_values, X]
    #print(f"SHAP Values: {shap_values.values}")

def model(train_data, file_name):
    print ("---------------------------------------------------------\n", file_name)
    #print(train_data.head(), "\n", train_data.size)
    train_data['time'] = train_data['time'].astype(str)
    train_data['time'] = train_data['time'].apply(lambda x: int(x.split(':')[0]))
    train_data['time'] = train_data['time'].astype(int)

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

    for column in columns_to_clean:
        train_data[column] = train_data[column].astype(str)
        train_data[column] = train_data[column].str.replace(',', '').astype(float)
        train_data[column] = pd.to_numeric(train_data[column], errors='coerce')

    X = train_data.drop(['region', 'price_demand_real', 'price_supply_real', 'price_demand', 'price_supply', 'date'],
                        axis=1)
    y = train_data['price_demand_real']

    # print(y.value_counts())
    # print(X.dtypes)  # Viewing the data types of features in X
    # print(X.isnull().sum())  # Check for any null values in X

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

    # Обучим модель на полных данных
    gbdt_reg = HistGradientBoostingRegressor(min_samples_leaf=10,
                                             max_depth=15,
                                             max_iter=250,
                                             learning_rate=0.1,
                                             random_state=42).fit(X, y)
    # по результатам перебора Best Parameters: {'learning_rate': 0.1, 'max_depth': 15, 'max_iter': 250, 'min_samples_leaf': 10}; Best Cross-Validated MSE Score: 25612.58657188177; param_grid = {'max_depth': [5, 10, 15, 20, 25, 30],'min_samples_leaf': [1, 2, 5, 10, 15, 20],'max_iter': [50, 100, 125, 150, 200, 250],'learning_rate': [0.01, 0.05, 0.1, 0.2]}

    y_pred = gbdt_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')


project_dir = os.getcwd()  # or replace 'os.getcwd()' with the specific path if needed

existing_file_names = ("irkutsk.csv", "moscow.csv", "tatarstan.csv", "chelyabinsk.csv")
print("Choose the train dataset\n", existing_file_names)
dataset_num = int(input("Input a number from 0 to 3:\n"))
# Define the path to the 'irkutsk.csv' file
file_name = existing_file_names[dataset_num]
filepath = os.path.join(project_dir, file_name)
train_data = pd.read_csv(filepath)


output = shap_statistics_on_model(train_data,file_name)
shap_values = output [0]
X = output [1]

#to_save_or_not_to_save = int(input("Do you want to save the shap statistics for visualization? \n 1 - yes, other - no:\n"))
#if to_save_or_not_to_save == 1:
#    Save_shap_values_and_X(shap_values, X)

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import shap
import os

def Summary_plot (shap_values, X, file_name):
    # 1. Summary Plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(f'{file_name}shap_summary_plot.png', bbox_inches='tight')  # Optional: Save the summary plot
    plt.show()  # Show the plot
    plt.close()

def Bar_plot(shap_values, X, file_name):
    # 2. Bar Plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig(f'{file_name}shap_summary_bar_plot.png')  # Optional: Save the bar plot
    plt.show()
    plt.close()

def Dependence_plot(shap_values, X, file_name, feature_num = 0):
    # 3. Dependence Plot
    # Choose a feature (for example, the first feature)
    shap.dependence_plot(feature_num, shap_values.values, X, show=False)  # Here, '0' is the index of the feature
    plt.savefig(f'{file_name}shap_dependence_plot_by_{X.columns[feature_num]}_feature_{feature_num}.png', bbox_inches='tight')  # Optional: Save the plot
    #plt.show()  # Show the plot
    #plt.close()


print(X.shape)

Summary_plot(shap_values, X, file_name[:-4])
Bar_plot(shap_values, X, file_name[:-4])
for num in range (0, 40):
    Dependence_plot(shap_values, X, file_name[:-4], num)
