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

    return np.array(selected_rows)

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
