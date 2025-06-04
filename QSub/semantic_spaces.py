import numpy as np
import requests
import re
import html
import concurrent.futures
from tqdm import tqdm

#############################################
## GALLITO BASED SEMANTIC SPACE OPERATIONS ##
#############################################

def get_word_vector_gallito(word, gallito_code, space_name):
    """
    Extrae el vector semántico de un término específico desde un espacio semántico
    proporcionado, utilizando el servicio web de Gallito.

    :param word: El término objetivo para el cual se busca el vector semántico.
    :type word: str
    :param gallito_code: Código de identificación para acceder al espacio semántico de Gallito.
    :type gallito_code: str
    :param space_name: Nombre del espacio semántico dentro del servicio de Gallito.
    :type space_name: str
    :return: Un array numpy que representa el vector semántico del término dado.
    :rtype: numpy.ndarray

    La función realiza una solicitud GET al servicio web de Gallito para obtener el
    vector del término especificado. Los valores del vector se extraen de la respuesta
    XML, se decodifican las entidades HTML y se convierten los valores numéricos a float.
    """
    # Extraer el vector del término objetivo
    resp = requests.get(
        f"http://comcog.psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={word}", timeout=10)
    content = resp.text
    # Decodificar las entidades HTML
    decoded_content = html.unescape(content)
    # Extraer todos los valores numéricos de las etiquetas <dim>
    vector_values = re.findall(r'<dim>(.*?)</dim>', decoded_content)
    # Convertir los valores a float y reemplazar comas por puntos
    word_vector = [float(value.replace(',', '.')) for value in vector_values]

    return np.array(word_vector)


def get_lsa_corpus_gallito(terms_file, gallito_code, space_name):
    """
    Lee una lista de términos de un archivo .txt, obtiene el vector de cada término
    usando la función 'word_vector', y guarda todo el corpus en una lista de 1darrays.
    También guarda en una lista todos los términos. Devuelve una tupla con dos listas

    Advertencia: Esta función es de una elevada exigencia computacional. Si se dispone del
    vector del supertérmino y de los cosenos del supertérmino ya ordenados, se recomienda
    no hacer uso de la misma.

    :param terms_file: Ruta al archivo .txt que contiene los términos.
    :type terms_file: str
    :param gallito_code: Código de acceso para la API de Gallito.
    :type gallito_code: str
    :param space_name: Nombre del espacio semántico en la API de Gallito.
    :type space_name: str
    :return: Un diccionario con los terminos en las keys y los vectores en los values.
    :rtype: dict
    """

    # Leer el archivo de términos
    with open(terms_file, 'r', encoding='utf-8') as file:
        individual_terms = [line.strip() for line in file]

    # Función auxiliar para usar en paralelo
    def get_term_vector(term):
        return term, get_word_vector_gallito(term, gallito_code, space_name)

    # Realizar solicitudes en paralelo y construir un diccionario
    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_term_vector, individual_terms),
                                 total=len(individual_terms),
                                 desc="Procesando términos"):
            corpus_dict[term] = np.array(vector)

    return corpus_dict

#######################################
## GENERAL SEMANTIC SPACE OPERATIONS ##
#######################################

def word_cosine_similarity(v1, v2):
    """
    Calcula la similitud coseno entre dos vectores.

    :param v1: Primer vector para el cálculo de la similitud coseno.
    :type v1: numpy.ndarray
    :param v2: Segundo vector para el cálculo de la similitud coseno.
    :type v2: numpy.ndarray
    :return: El valor de la similitud coseno entre los dos vectores.
    :rtype: float

    Esta función calcula la similitud coseno, una medida de similitud entre dos vectores
    no nulos en un espacio que tiene en cuenta la orientación de los vectores pero no su magnitud.
    La similitud coseno se define como el producto punto de los vectores dividido por el producto
    de sus magnitudes (normas).
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    # Evitar divisiones por cero si alguno de los vectores es nulo
    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_sim = np.dot(v1, v2) / (norm1 * norm2)
    return cos_sim
