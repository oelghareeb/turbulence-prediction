import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load the combined data
def load_combined_array(cases, field, dataset='komegasst'):
    data = np.concatenate([np.load(f'/kaggle/input/ml-turbulence-dataset/{dataset}/{dataset}_{case}_{field}.npy') for case in cases])
    return data

# Preprocessing the data
def preprocess_data(x, y, feature_columns, label_column):
    scaler_features = MinMaxScaler()
    scaler_labels = MinMaxScaler()

    # Scale the features
    x_scaled = scaler_features.fit_transform(x)
    y_scaled = scaler_labels.fit_transform(y.values.reshape(-1, 1))

    # Return the scaled data
    return x_scaled, y_scaled, scaler_features, scaler_labels
