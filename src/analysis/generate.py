import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_symmetric_points(num_points, seed=None):
    """
    Generate symmetric points for given angle and eccentricity ranges.
    
    Parameters:
    num_points (int): Number of points to generate (each will have a symmetric counterpart).
    
    Returns:
    DataFrame: A pandas DataFrame with columns 'angle' and 'eccentricity'.

    """

    if seed is not None:
        np.random.seed(seed)

    angles = np.random.uniform(0, 360, num_points)
    eccentricities = np.random.uniform(0, 20, num_points)
    
    symmetric_angles = -angles
    symmetric_eccentricities = -eccentricities
    
    angles = np.concatenate((angles, symmetric_angles))
    eccentricities = np.concatenate((eccentricities, symmetric_eccentricities))
    
    data = {
        'angle': angles,
        'eccentricity': eccentricities
    }
    
    return pd.DataFrame(data)

def generate_circular_points(num_angles, eccentricities):
    """
    Generate points distributed along circles for different eccentricities.
    
    Parameters:
    num_angles (int): Number of angles to generate per circle.
    eccentricities (list): List of eccentricity values for the circles.
    
    Returns:
    DataFrame: A pandas DataFrame with columns 'angle' and 'eccentricity'.
    """
    angles = np.linspace(0, 360, num_angles, endpoint=False)
    data = {
        'angle': [],
        'eccentricity': []
    }
    
    for ecc in eccentricities:
        data['angle'].extend(angles)
        data['eccentricity'].extend([ecc] * num_angles)
    
    return pd.DataFrame(data)

# Parameters
num_angles = 36  # Number of points per circle
eccentricities = [2, 4, 6, 8, 10]  # List of eccentricities

# Generate the dataset
dataset = generate_circular_points(num_angles, eccentricities)


def plot_symmetric_points(dataset):
    """
    Plot symmetric points from the dataset.
    
    Parameters:
    dataset (DataFrame): A pandas DataFrame with columns 'angle' and 'eccentricity'.
    """
    angles = dataset['angle']
    eccentricities = dataset['eccentricity']
    
    # Convert polar coordinates to Cartesian for plotting
    x = eccentricities * np.cos(np.radians(angles))
    y = eccentricities * np.sin(np.radians(angles))
    
    plt.figure(figsize=(8, 8))
    plt.scatter(x, y, c='blue', label='Points')
    
    plt.title('Symmetric Points Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(True)
    plt.legend()
    plt.show()

# Generate the dataset
# num_points = 100  # You can set this to any number you prefer
# eccentricities = np.arange(0,20)

# dataset = generate_circular_points(num_points, eccentricities)

# plot_symmetric_points(dataset)


