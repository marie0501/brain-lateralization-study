from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, Matern, ConstantKernel as C
import pandas as pd
from generate import generate_circular_points
import numpy as np
import matplotlib.pyplot as plt


def train_gaussian_process(train_data, kernel):
    """
    Train a Gaussian Process model using the training data.
    
    Parameters:
    train_data (DataFrame): A pandas DataFrame with columns 'angle', 'eccentricity', and 'value'.
    
    Returns:
    GaussianProcessRegressor: The trained Gaussian Process model.
    """
    # Extract input features and target values
    X_train = train_data[['angle', 'eccentricity']].values
    y_train = train_data['size'].values

    # Create GaussianProcessRegressor model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

    # Fit to the training data
    gp.fit(X_train, y_train)
    
    return gp

def predict_gaussian_process(gp_model, dataset):
    """
    Predict values using the trained Gaussian Process model.
    
    Parameters:
    gp_model (GaussianProcessRegressor): The trained Gaussian Process model.
    dataset (DataFrame): A pandas DataFrame with columns 'angle' and 'eccentricity'.
    
    Returns:
    DataFrame: The input dataset with an additional 'predicted_value' column.
    """
    # Extract input features
    X_pred = dataset[['angle', 'eccentricity']].values
    
    # Predict values
    predicted_values, std_devs = gp_model.predict(X_pred, return_std=True)
    
    # Add predictions to the dataset
    dataset['predicted_size'] = predicted_values
    dataset['std_dev'] = std_devs
    
    return dataset


def plot_points(train_data, predicted_dataset):
    angles_train = train_data['angle']
    ecc_train = train_data['eccentricity']
    values_train = train_data['size']
    
    x_train = ecc_train * np.cos(np.radians(angles_train))
    y_train = ecc_train * np.sin(np.radians(angles_train))
    
    angles_pred = predicted_dataset['angle']
    ecc_pred = predicted_dataset['eccentricity']
    values_pred = predicted_dataset['predicted_size']
    
    x_pred = ecc_pred * np.cos(np.radians(angles_pred))
    y_pred = ecc_pred * np.sin(np.radians(angles_pred))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(x_train, y_train, c=values_train, cmap='viridis', label='Original Points')
    plt.colorbar(scatter, label='Size')
    plt.title('Original Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(x_pred, y_pred, c=values_pred, cmap='viridis', label='Predicted Points')
    plt.colorbar(scatter, label='Predicted Size')
    plt.title('Predicted Points')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


# Kernels

rbf_kernel = C(1.0) * RBF(length_scale=1.0)
rational_quadratic_kernel = C(1.0) * RationalQuadratic(length_scale=1.0, alpha=0.5)
matern_kernel = C(1.0) * Matern(length_scale=1.0, nu=1.5)


train_data = pd.read_csv('/Users/mariedelvalle/Downloads/HCP_hV4.csv')
train_data_cleaned = train_data.dropna(how='any')

# train_data_cleaned = train_data_cleaned[train_data_cleaned['eccentricity'] < 3]
train_data_cleaned = train_data_cleaned[train_data_cleaned['R2'] > 20]

subjects = train_data_cleaned['subj'].unique()

for sub in subjects[:10]:

    train_data_sub = train_data_cleaned[train_data_cleaned['subj']==sub]
    print(f"Subject: {sub} - {len(train_data_sub)}")

    num_points = 100
    seed = 42
    dataset = generate_circular_points(num_points, np.arange(0,20))

    # Train the Gaussian Process model
    kernel = matern_kernel
    gp_model = train_gaussian_process(train_data_sub, kernel)

    # Predict values for the generated points
    predicted_dataset = predict_gaussian_process(gp_model, dataset)

    # Plot the original and predicted points
    plot_points(train_data_cleaned, predicted_dataset)