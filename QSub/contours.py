import requests
import re
import numpy as np
import xml.etree.ElementTree as ET
import html
from numpy.linalg import norm


def gallito_neighbors_matrix(word, gallito_code, space_name, neighbors=100, min_cosine_contour=0.3, space_dimensions = 300):
    k = space_dimensions  # Define K dimensions
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

    space_url = f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getNearestNeighboursList"
    response = requests.post(space_url, data=query_xml, headers={'Content-Type': 'text/xml'})

    # Decodificar las entidades HTML
    decoded_content = html.unescape(response.text)
    namespaces = {'ns': 'http://tempuri.org/'}

    # Procesar la respuesta decodificada
    root = ET.fromstring(decoded_content)
    # Ajustar la consulta XPath para incluir el espacio de nombres
    terms = [term.text for term in root.findall('.//ns:item/ns:term', namespaces)]

    if not terms:
        print("No se encontraron términos.")
        return None

    matrix = np.zeros((k, len(terms)))

    for i, term in enumerate(terms):
        vector_url = f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={term}"
        vector_response = requests.get(vector_url)

        # Decodificar las entidades HTML
        decoded_vector_content = html.unescape(vector_response.text)

        # Extraer todos los valores numéricos de las etiquetas <dim>
        vector_values = re.findall(r'<dim>(.*?)</dim>', decoded_vector_content)

        # Convertir los valores a float y reemplazar comas por puntos
        vector = np.array([float(value.replace(',', '.')) for value in vector_values])

        # Añadir el vector a la matriz
        matrix[:, i] = vector

    # Extraer el vector del término objetivo
    resp_a = requests.get(
        f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={word}")
    content = resp_a.text

    # Decodificar las entidades HTML
    decoded_content = html.unescape(content)

    # Extraer todos los valores numéricos de las etiquetas <dim>
    vector_values = re.findall(r'<dim>(.*?)</dim>', decoded_content)

    # Convertir los valores a float y reemplazar comas por puntos
    word_vector = [float(value.replace(',', '.')) for value in vector_values]

    # Calcular la correlación y filtrar por el mínimo coseno
    selected_rows = []

    for row in matrix.T:
        if np.dot(row, word_vector) / (norm(row) * norm(word_vector)) > 0.3:
            selected_rows.append(row)

    results = {"neighbors": terms, "neighbors_vec": np.array(selected_rows)}

    return results

def word_vector(word, gallito_code, space_name):
    # Extraer el vector del término objetivo
    resp_a = requests.get(
        f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={word}")
    content = resp_a.text
    # Decodificar las entidades HTML
    decoded_content = html.unescape(content)
    # Extraer todos los valores numéricos de las etiquetas <dim>
    vector_values = re.findall(r'<dim>(.*?)</dim>', decoded_content)
    # Convertir los valores a float y reemplazar comas por puntos
    word_vector = [float(value.replace(',', '.')) for value in vector_values]

    return np.array(word_vector)


def word_cosine_similarity(v1, v2):
    """Calcula la similitud coseno entre dos vectores."""
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return cos_sim


def neighbors_similarity(word_semantic_vector, word_neighbors_matrix):
    """
    Calcula la similitud coseno entre un vector semántico de una palabra y cada vector
    en una matriz de vecinos, ordenando los resultados en orden descendente. También devuelve
    los términos asociados en el mismo orden.

    :param word_semantic_vector: ndarray, el vector semántico de la palabra objetivo.
    :param word_neighbors_matrix: dict, con "neighbors_vec" (ndarray) y "neighbors" (lista de términos).
    :return: Tuple de dos ndarrays, el primero de similitudes coseno y el segundo de términos, ambos ordenados.
    """
    cosine_vec = np.zeros(word_neighbors_matrix["neighbors_vec"].shape[0])
    terms = word_neighbors_matrix["neighbors"]

    for i in range(word_neighbors_matrix["neighbors_vec"].shape[0]):
        cosine_vec[i] = word_cosine_similarity(word_neighbors_matrix["neighbors_vec"][i, :], word_semantic_vector)

    # Crea un array estructurado para mantener juntos los cosenos y los términos
    combined = np.array(list(zip(cosine_vec, terms)), dtype=[('cosine', 'f4'), ('term', 'U20')])

    # Ordena el array basado en las similitudes coseno
    combined_sorted = np.sort(combined, order='cosine')[::-1]

    # Separa y devuelve los vectores de cosenos y términos
    sorted_cosines = combined_sorted['cosine']
    sorted_terms = combined_sorted['term']
    return sorted_cosines, sorted_terms

