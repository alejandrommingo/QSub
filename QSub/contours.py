import requests
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fuzzywuzzy import process
import xml.etree.ElementTree as ET
import html
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from numpy.linalg import norm
from QSub.semantic_spaces import get_word_vector_gallito, word_cosine_similarity

# In this module you can find all functions related to the creation of contextual and conceptual contours

######################################
### GALLITO API BASED CONTORUS LSA ###
######################################

def get_neighbors_matrix_gallito(word, gallito_code, space_name, neighbors=100):
    """
    Consulta un espacio semántico específico (denominado 'Gallito') para obtener
    una matriz de los vecinos semánticos más cercanos a un término dado. Extrae los
    vectores semánticos de los términos vecinos y filtra estos vecinos basándose
    en una similitud coseno mínima especificada.

    :param word: El término objetivo para el cual se buscan vecinos semánticos.
    :type word: str
    :param gallito_code: Código de identificación para acceder al espacio semántico de Gallito.
    :type gallito_code: str
    :param space_name: Nombre del espacio semántico dentro del servicio de Gallito.
    :type space_name: str
    :param neighbors: Número de vecinos semánticos a recuperar. Por defecto es 100.
    :type neighbors: int
    :return: un diccionario con los términos en las keys y los vectores semánticos en los values.
    :rtype: dict
    """
    n = neighbors  # Define N neighbors

    # Preparar la consulta a la API de Gallito
    query_xml = f"""
            <getNearestNeighboursList xmlns='http://tempuri.org/'>
                <code>{gallito_code}</code>
                <a>{word}</a>
                <txtNegate></txtNegate>
                <n>{n}</n>
                <lenght_biased>false</lenght_biased>
            </getNearestNeighboursList>
            """

    space_url = f"http://comcog.psicoee.uned.es/{space_name}/Service.svc/webHttp/getNearestNeighboursList"
    response = requests.post(space_url, data=query_xml, headers={'Content-Type': 'text/xml'}, timeout=10)

    decoded_content = html.unescape(response.text)
    namespaces = {'ns': 'http://tempuri.org/'}
    root = ET.fromstring(decoded_content)
    terms = [term.text for term in root.findall('.//ns:item/ns:term', namespaces)]

    if not terms:
        print("No se encontraron términos.")
        return None

    # Función para obtener el vector de un término
    def get_vector(term):
        vector_url = f"http://comcog.psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={term}"
        vector_response = requests.get(vector_url, timeout=10)
        decoded_vector_content = html.unescape(vector_response.text)
        vector_values = re.findall(r'<dim>(.*?)</dim>', decoded_vector_content)
        vector = np.array([float(value.replace(',', '.')) for value in vector_values])
        # Normalizar el vector
        normalized_vector = vector / norm(vector)
        return normalized_vector

    # Obtener vectores de términos en paralelo
    with ThreadPoolExecutor() as executor:
        term_vectors = list(tqdm(executor.map(get_vector, terms), total=len(terms), desc="Obteniendo vectores"))

    # Crear diccionario de términos y vectores
    neighbors_dict = {term: vec for term, vec in zip(terms, term_vectors)}

    return neighbors_dict

def get_superterm_gallito(terms_file, gallito_code, space_name):
    """
    Lee una lista de términos de un archivo .txt, obtiene el vector de cada término
    usando la función 'word_vector', suma estos vectores y devuelve el resultado y un vector
    de similitudes coseno del vector suma con cada término.

    Advertencia: Esta función es de una elevada exigencia computacional. Si se dispone del
    vector del supertérmino y de los cosenos del supertérmino ya ordenados, se recomienda
    no hacer uso de la misma.

    :param terms_file: Ruta al archivo .txt que contiene los términos.
    :type terms_file: str
    :param gallito_code: Código de acceso para la API de Gallito.
    :type gallito_code: str
    :param space_name: Nombre del espacio semántico en la API de Gallito.
    :type space_name: str
    :return: Una tupla que contiene dos elementos: el primer elemento es el vector suma de todos los
             vectores de términos, y el segundo elemento es un array numpy de similitudes coseno entre
             el vector suma y cada uno de los vectores individuales de términos.
    :rtype: tuple

    La función lee los términos desde el archivo especificado, utiliza 'word_vector' para obtener el vector
    semántico de cada término y los suma para formar el vector del supertérmino. Luego, calcula la similitud
    coseno entre este vector suma y cada uno de los vectores de términos individuales y devuelve ambos, el
    vector suma y las similitudes coseno, en una tupla.
    """

    # Inicializa el vector sumatorio y la lista de vectores individuales
    superterm_vector = None
    individual_vectors = []

    # Leer el archivo de términos
    with open(terms_file, 'r', encoding='utf-8') as file:
        for line in file:
            term = line.strip()  # Elimina espacios y saltos de línea
            term_vector = get_word_vector_gallito(term, gallito_code, space_name)
            individual_vectors.append(term_vector)

            # Sumar vectores
            if superterm_vector is None:
                superterm_vector = term_vector
            else:
                superterm_vector += term_vector

    # Calcular las similitudes coseno
    cosine_similarities = np.array([word_cosine_similarity(superterm_vector, v) for v in individual_vectors])

    return superterm_vector, cosine_similarities

###################################
## CONTOURS EVALUATION FUNCTIONS ##
###################################
def find_closest_neighbors_lsa(word, lsa_corpus, n_neighbors=10):
    """
    Encuentra los n_neighbors más cercanos a una palabra dada en un corpus LSA.

    :param word: La palabra objetivo para encontrar vecinos.
    :type word: str
    :param lsa_corpus: Un diccionario que contiene palabras y sus vectores semánticos.
    :type lsa_corpus: dict
    :param n_neighbors: Número de vecinos más cercanos a devolver.
    :type n_neighbors: int
    :return: Un diccionario con los términos vecinos como claves y sus vectores semánticos como valores.
    :rtype: dict
    """

    if word not in lsa_corpus:
        raise ValueError(f"La palabra '{word}' no se encuentra en el corpus.")

    # Vector de la palabra objetivo
    word_vector = lsa_corpus[word]

    # Preparar los vectores del corpus para el cálculo vectorizado, excluyendo el vector de la palabra objetivo
    corpus_terms, corpus_vectors = zip(*[(term, vector) for term, vector in lsa_corpus.items() if term != word])
    corpus_matrix = np.stack(corpus_vectors)

    # Calcular similitudes coseno de manera vectorizada
    dot_product = np.dot(corpus_matrix, word_vector)
    norm_product = np.linalg.norm(word_vector) * np.linalg.norm(corpus_matrix, axis=1)
    similarities = dot_product / norm_product

    # Ordenar los resultados y tomar los top n_neighbors
    sorted_indices = np.argsort(-similarities)[:n_neighbors]
    closest_neighbors_dict = {corpus_terms[i]: lsa_corpus[corpus_terms[i]] for i in sorted_indices}

    return closest_neighbors_dict

def neighbors_similarity(word_semantic_vector, word_neighbors_dict):
    """
    Calcula la similitud coseno entre un vector semántico de una palabra y cada vector
    en una matriz de vecinos, ordenando los resultados en orden descendente. También devuelve
    los términos asociados en el mismo orden.

    :param word_semantic_vector: El vector semántico de la palabra objetivo.
    :type word_semantic_vector: numpy.ndarray
    :param word_neighbors_dict: Un diccionario que contiene dos elementos:
                                  "neighbors_vec" que es un ndarray con los vectores de los vecinos semánticos,
                                  y "neighbors" que es una lista de los términos correspondientes a esos vecores.
    :type word_neighbors_dict: dict
    :return: Una tupla de dos ndarrays. El primero contiene las similitudes coseno ordenadas en orden descendente,
             y el segundo contiene los términos asociados ordenados en concordancia con las similitudes coseno.
    :rtype: tuple

    Esta función primero calcula la similitud coseno entre el vector semántico de una palabra y cada uno de
    los vectores en una matriz de vecinos. Luego, crea un array estructurado que combina estas similitudes
    con los términos correspondientes, ordena este array por similitud coseno en orden descendente, y finalmente
    separa y devuelve los vectores de similitudes coseno y términos en este orden.
    """
    # Calcular la similitud coseno para cada vecino en el diccionario
    cosine_similarities = {}
    for term, neighbor_vector in word_neighbors_dict.items():
        cosine_similarity = word_cosine_similarity(neighbor_vector, word_semantic_vector)
        cosine_similarities[term] = cosine_similarity

    # Ordenar el diccionario basado en las similitudes coseno en orden descendente
    sorted_cosine_similarities = dict(sorted(cosine_similarities.items(), key=lambda item: item[1], reverse=True))

    return sorted_cosine_similarities


def deserved_neighbors(word, df_h_values, sorted_cos, word_cosines, graph=False):
    """
    Analiza y compara la similitud coseno de una palabra objetivo con un 'superterm' y
    sus vecinos más cercanos.

    :param word: Palabra objetivo para la comparación.
    :type word: str
    :param df_h_values: DataFrame previamente cargado que contiene los valores de diversidad
                        semántica ('h' y 'h_dec') para las palabras.
    :type df_h_values: pandas.DataFrame
    :param sorted_cos: Array de NumPy con los valores de similitud coseno del 'superterm',
                       resultante de la función get_superterm (segundo elemento de la tupla devuelta).
    :type sorted_cos: numpy.ndarray
    :param word_cosines: Array de NumPy con los valores de similitud coseno de la palabra objetivo
                         con sus vecinos, resultante de la función neighbors_similarity (elemento
                         'sorted_cosines' de la tupla devuelta).
    :type word_cosines: numpy.ndarray
    :param graph: Si es True, genera un gráfico del análisis paralelo, por defecto False.
    :type graph: bool
    :return: El número de vecinos cuya similitud con la palabra objetivo es mayor o igual que la
             similitud con el 'superterm'.
    :rtype: int

    La función calcula el peso del 'superterm' ajustado por la diversidad semántica de la palabra
    objetivo, compara las similitudes coseno y genera un gráfico para visualizar la comparación.
    """

    # Calcular h_dec para la palabra objetivo
    if word in df_h_values['word'].values:
        h_dec = df_h_values[df_h_values['word'] == word]['h_dec'].iloc[0]
    else:
        # Encontrar la palabra más similar en df_h_values
        closest_word, similarity_score = process.extractOne(word, df_h_values['word'].tolist())
        h_dec = df_h_values[df_h_values['word'] == closest_word]['h_dec'].iloc[0]

        # Aviso de que se ha usado una palabra similar
        print(f"Advertencia: La palabra '{word}' no se encontró en el DataFrame. Usando '{closest_word}' (puntuación de similitud: {similarity_score}) como la más similar.")

    # Calcular el superterm ponderado
    superterm_weighted = sorted_cos / (1 + h_dec / 10)

    # Aplanar los arrays para asegurar que son unidimensionales
    sorted_cos = sorted_cos.flatten()
    superterm_weighted = superterm_weighted.flatten()
    word_cosines = word_cosines.flatten()

    # Crear el DataFrame para el análisis
    data = pd.DataFrame({
        "superterm": sorted_cos[:len(word_cosines)],
        "superterm_weighted": superterm_weighted[:len(word_cosines)],
        "word_cosines": word_cosines,
        "secuencia": np.arange(1, len(word_cosines) + 1)
    })

    sum_value = sum(data['word_cosines'] >= data['superterm_weighted'])

    # Generar el gráfico
    if graph:
        plt.figure(figsize=(10, 6))
        plt.plot(data['secuencia'], data['superterm_weighted'], label='Weighted Superterm')
        plt.plot(data['secuencia'], data['word_cosines'], label=word)
        plt.xlabel('Number of Neighbors')
        plt.ylabel('Similarity')
        plt.title(f'Superterm vs {word.capitalize()} Cosine Similarities')
        plt.legend()
        plt.show()

    return sum_value
