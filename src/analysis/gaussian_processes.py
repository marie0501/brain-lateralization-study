def fit_gaussian(x_data, y_data):
    popt, pcov = curve_fit(gaussian, x_data, y_data)
    return popt

def generate_smooth_distribution(x_data, popt):
    x_smooth = np.linspace(min(x_data), max(x_data), 1000)
    y_smooth = gaussian(x_smooth, *popt)
    return x_smooth, y_smooth

def plot_results(x_data, y_data, x_smooth, y_smooth):
    plt.scatter(x_data, y_data, label='Datos Originales')
    plt.plot(x_smooth, y_smooth, 'r-', label='Distribución Suavizada')
    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('Ajuste de Distribución Gaussiana a los Datos de PRF')
    plt.legend()
    plt.grid(True)
    plt.show()

# Aplicar el proceso gaussiano a los datos de PRF por sujeto
popt = fit_gaussian(x_data, y_data)
x_smooth, y_smooth = generate_smooth_distribution(x_data, popt)
plot_results(x_data, y_data, x_smooth, y_smooth)
