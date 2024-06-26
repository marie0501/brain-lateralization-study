import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def polar_to_cartesian(ecc, ang):
    """
    Convierte coordenadas polares a cartesianas.
    
    Parameters:
    ecc (float): Excentricidad (radio).
    ang (float): Ángulo polar en grados.
    
    Returns:
    tuple: Coordenadas cartesianas (x, y).
    """
    ang_rad = np.deg2rad(ang)  # Convertir ángulo de grados a radianes
    x = ecc * np.cos(ang_rad)  # Coordenada X
    y = ecc * np.sin(ang_rad)  # Coordenada Y
    return x, y

def add_cartesian_coordinates(df):
    """
    Añade columnas de coordenadas cartesianas (x, y) al DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame con columnas 'ecc' y 'ang'.
    
    Returns:
    DataFrame: DataFrame con columnas adicionales 'x' y 'y'.
    """
    cartesian_coords = df.apply(lambda row: polar_to_cartesian(row['ecc'], row['ang']), axis=1)
    df[['x', 'y']] = pd.DataFrame(cartesian_coords.tolist(), index=df.index)
    return df


def plot_cartesian_coordinates(df):
    """
    Grafica las coordenadas cartesianas del DataFrame.
    
    Parameters:
    df (DataFrame): DataFrame con columnas 'x', 'y' y 'hemisphere'.
    """
    plt.figure(figsize=(10, 6))
    for hemisphere in df['hemisphere'].unique():
        subset = df[df['hemisphere'] == hemisphere]
        plt.scatter(subset['x'], subset['y'], label=hemisphere)
    
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Representación de Campos Receptivos en Coordenadas Cartesianas')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

def main():
    # Supongamos que df es el DataFrame que contiene tus datos con columnas 'ecc' y 'ang'
    # Crear un DataFrame de ejemplo
    data = {
        'subject_id': ['sub1', 'sub2'],
        'hemisphere': ['lh', 'rh'],
        'ecc': [10, 20],
        'ang': [30, 60]
    }
    df = pd.DataFrame(data)
    
    # Añadir coordenadas cartesianas al DataFrame
    df = add_cartesian_coordinates(df)
    
    # Graficar los datos
    plot_cartesian_coordinates(df)

if __name__ == "__main__":
    main()

