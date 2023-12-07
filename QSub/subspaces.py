import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from numpy.linalg import norm
from wordcloud import WordCloud

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
    for _ in tqdm(range(num_simulations), desc="Simulando datos aleatorios"):
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

def create_subspace(word_contour, n_comp, subspace_type="Terms"):
    """
    Crea un subespacio de componentes principales a partir de los contornos de una palabra,
    utilizando el Análisis de Componentes Principales (PCA) para reducir la dimensionalidad.

    :param word_contour: Contorno de la palabra representado como un diccionario de términos
                         y sus vectores semánticos o como una matriz de vectores semánticos.
    :type word_contour: dict or numpy.ndarray
    :param n_comp: Número de componentes principales a extraer.
    :type n_comp: int
    :param subspace_type: Tipo de subespacio a crear. 'Terms' para aplicar PCA en los términos (filas),
                          cualquier otro valor para aplicar PCA en las características (columnas).
                          Por defecto es 'Terms'.
    :type subspace_type: str

    :return: Matriz de componentes principales normalizados en la dimensión original de los vectores.
    :rtype: numpy.ndarray

    Esta función aplica PCA para reducir la dimensionalidad de los vectores semánticos. En el caso
    de 'Terms', PCA se aplica a las filas de la matriz de contornos (términos), y en otros casos,
    se aplica a las columnas (características). Los componentes resultantes se normalizan usando la
    norma tipo 2 para mantener la coherencia en la magnitud de los vectores.
    
    Ejemplo de uso:
        subspace = create_subspace(word_contour_dict, 4, subspace_type='Terms')
        # Esto devuelve una matriz de 4 componentes normalizados en la dimensión original.
    """
    if isinstance(word_contour, dict):
        word_contour = np.array(list(word_contour.values()))

    if subspace_type == 'Terms':
        word_contour = word_contour.T

    pca = PCA(n_components=n_comp)
    pca.fit(word_contour)

    if subspace_type == 'Terms':
        components = np.dot(pca.components_, word_contour.T)
    else:
        components = pca.components_

    # Normalizar los componentes usando la norma tipo 2
    normalized_components = components / norm(components, axis=1, keepdims=True)

    return normalized_components

def describe_subspace(subespacio_matrix, contour, top_n=10, graph = False):
    """
    Describe un subespacio calculando la similitud coseno de cada término en el contorno con
    cada dimensión del subespacio y devuelve los términos más similares para cada dimensión.

    :param subespacio_matrix: Matriz que representa el subespacio, donde cada columna es una dimensión.
    :type subespacio_matrix: numpy.ndarray
    :param contour: Contorno de términos representado como un diccionario, donde las keys son términos
                    y los values son sus vectores semánticos.
    :type contour: dict
    :param top_n: Número de términos más similares a retornar para cada dimensión del subespacio.
                  Por defecto es 10.
    :type top_n: int
    :param graph: Si es True, genera los wordclouds, por defecto False.
    :type graph: bool

    :return: Un diccionario donde cada key representa una dimensión del subespacio y cada value es otro
             diccionario con los top_n términos más similares y sus respectivas similitudes coseno.
    :rtype: dict

    Esta función primero convierte el contorno de términos en una matriz y luego calcula la similitud
    coseno entre cada término y cada dimensión del subespacio. Finalmente, selecciona y devuelve los
    top_n términos más similares para cada dimensión.

    Ejemplo de uso:
        resultados = describe_subspace(subespacio_matrix, contour_dict, top_n=5)
        # Esto devuelve los 5 términos más similares a cada dimensión del subespacio junto con sus similitudes.
    """

    contour_m = np.array(list(contour.values()))
    terminos_lista = list(contour.keys())
    subespacio_cosines_matrix = np.dot(subespacio_matrix, contour_m.T).T

    resultados = {}

    # Iterar a través de cada columna (dimensión) de la matriz
    for i in range(subespacio_cosines_matrix.shape[1]):
        top_indices = np.argsort(subespacio_cosines_matrix[:, i])[::-1][:top_n]
        top_terminos = {terminos_lista[idx]: subespacio_cosines_matrix[idx, i] for idx in top_indices}
        resultados[f"Dimension_{i+1}"] = top_terminos
    
    if graph:
        for dimension, terminos in resultados.items():
            wordcloud = WordCloud(width=800, height=400).generate_from_frequencies(terminos)

            # Mostrar la word cloud
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.title(dimension)
            plt.show()

    return resultados