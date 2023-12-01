import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

def parallel_analysis_horn(data_matrix, analysis_type='Terms', num_simulations=100, graph=False):
    """
    Realiza un análisis paralelo de Horn para determinar el número de componentes principales
    a retener de una matriz de datos, con la opción de enfocar el análisis en términos o características.

    :param data_matrix: Matriz de datos o diccionario de términos y vectores.
    :type data_matrix: numpy.ndarray or dict
    :param analysis_type: Especifica el tipo de subespacio a analizar: 'Terms' o 'Features'.
    :type analysis_type: str
    :param num_simulations: Número de simulaciones para generar datos aleatorios, por defecto 100.
    :type num_simulations: int
    :param graph: Si es True, genera un gráfico del análisis paralelo, por defecto False.
    :type graph: bool
    :return: Número de componentes a retener según el análisis paralelo de Horn.
    :rtype: int
    """

    # Convertir el diccionario de datos en una matriz numpy si es necesario
    if isinstance(data_matrix, dict):
        data_matrix = np.array(list(data_matrix.values()))
        if analysis_type == 'Terms':
            data_matrix = data_matrix.T

    # Normalizar la matriz de datos
    normalized_data_matrix = (data_matrix - np.mean(data_matrix, axis=0)) / np.std(data_matrix, axis=0)

    # Ejecutar PCA en la matriz normalizada
    pca = PCA(n_components=min(data_matrix.shape) - 1)
    pca.fit(normalized_data_matrix)

    # Generar datos aleatorios y calcular la media de sus valores propios
    random_eigenvalues = np.zeros(pca.n_components_)
    for i in tqdm(range(num_simulations), desc="Simulando datos aleatorios"):
        random_data = np.random.normal(0, 1, data_matrix.shape)
        pca_random = PCA(n_components=pca.n_components_)
        pca_random.fit(random_data)
        random_eigenvalues += pca_random.explained_variance_ratio_
    random_eigenvalues /= num_simulations

    # Comparar los valores propios reales con los promedios de los valores propios aleatorios
    num_components_to_retain = np.sum(pca.explained_variance_ratio_ > random_eigenvalues)

    # Generar gráfico, si se solicita
    if graph:
        plt.plot(pca.explained_variance_ratio_, '--bo', label='PCA de los datos reales')
        plt.plot(random_eigenvalues, '--rx', label='Media de PCA de datos aleatorios')
        plt.legend()
        plt.title('Gráfico de Análisis Paralelo')
        plt.xlabel('Número de Componente')
        plt.ylabel('Proporción de Varianza Explicada')
        plt.show()

    return num_components_to_retain
