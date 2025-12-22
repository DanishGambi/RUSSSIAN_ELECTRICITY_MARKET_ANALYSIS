# -------------------------------------------------------------------
# PROJECT: Анализ потребления электроэнергии в регионах России
# -------------------------------------------------------------------
# Copyright 2024 Dmitrii A. Maliuzhantsev
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pandas as pd
import shap
import os

def Summary_plot (shap_values, X):
    # 1. Summary Plot
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig('shap_summary_plot.png', bbox_inches='tight')  # Optional: Save the summary plot
    plt.show()  # Show the plot
    plt.close()

def Bar_plot(shap_values, X):
    # 2. Bar Plot
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.savefig('shap_summary_bar_plot.png')  # Optional: Save the bar plot
    plt.show()
    plt.close()

def Dependence_plot(shap_values, X, feature_num = 0):
    # 3. Dependence Plot
    # Choose a feature (for example, the first feature)
    shap.dependence_plot(feature_num, shap_values.values, X, show=False)  # Here, '0' is the index of the feature
    plt.savefig(f'shap_dependence_plot_by_{X.columns[feature_num]}_feature_{feature_num}.png', bbox_inches='tight')  # Optional: Save the plot
    #plt.show()  # Show the plot
    plt.close()

def Save_shap_values_and_X (shap_values, X, x_file='X_data.pkl', shap_file='shap_values.pkl'):
    import pickle
    with open(shap_file, 'wb') as f:
        pickle.dump(shap_values, f)

    original_data_df = pd.DataFrame(X)
    with open(x_file, 'wb') as f:
        pickle.dump(original_data_df, f)
    print ("shap_values_and_X saved successfully.")

def Load_shap_values_and_X (x_file='X_data.pkl', shap_file='shap_values.pkl'):
    import pickle
    with open('shap_values.pkl', 'rb') as f:
        loaded_shap_values = pickle.load(f)

    with open('X_data.pkl', 'rb') as f:
        loaded_X = pickle.load(f)

    print("shap_values_and_X loaded successfully.")

    return [loaded_shap_values, loaded_X]

Loaded_data = Load_shap_values_and_X()
shap_values = Loaded_data[0]
X = Loaded_data[1]
print(X.shape)

#Summary_plot(shap_values, X)
#Bar_plot(shap_values, X)
for num in range (0, 40):

    Dependence_plot(shap_values, X, num)

