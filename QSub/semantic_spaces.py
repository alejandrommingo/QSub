import numpy as np
import requests
import re
import html
import concurrent.futures
from tqdm import tqdm
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:  # pragma: no cover - optional dependency
    AutoTokenizer = None
    AutoModel = None

try:  # pragma: no cover - optional dependency
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:  # pragma: no cover - optional dependency
    from wordfreq import top_n_list
except ImportError:  # pragma: no cover - optional dependency
    top_n_list = None

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

#############################################
## BERT BASED SEMANTIC SPACE OPERATIONS    ##
#############################################

_bert_models = {}


def _load_bert(model_name):
    """Load tokenizer and model for ``model_name`` with hidden states."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return tokenizer, model


def get_word_vector_bert(word, model_name="bert-base-uncased", output_layer="last"):
    """Return BERT embeddings for a given word.

    Parameters
    ----------
    word : str
        Target word.
    model_name : str, optional
        HuggingFace model name.
    output_layer : str | int, optional
        ``"last"`` (default) returns the final layer. An integer selects the
        corresponding intermediate layer (0-indexed). ``"all"`` returns an array
        with one vector per layer.
    """
    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise ImportError(
            "transformers and torch must be installed to use get_word_vector_bert"
        )

    if model_name not in _bert_models:
        _bert_models[model_name] = _load_bert(model_name)

    tokenizer, model = _bert_models[model_name]
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    if output_layer == "last":
        vector = outputs.last_hidden_state.mean(dim=1).squeeze()
        return vector.numpy()

    if output_layer == "all":
        hidden = [h.mean(dim=1).squeeze().numpy() for h in outputs.hidden_states[1:]]
        return np.stack(hidden)

    if isinstance(output_layer, int):
        hidden_states = outputs.hidden_states[1:]
        if output_layer < 0 or output_layer >= len(hidden_states):
            raise ValueError("output_layer out of range")
        vector = hidden_states[output_layer].mean(dim=1).squeeze()
        return vector.numpy()

    raise ValueError("output_layer must be 'last', 'all', or an integer index")


def get_bert_corpus(
    language="en",
    model_name="bert-base-uncased",
    n_words=1000,
    output_layer="last",
):
    """Return a dictionary with BERT vectors of the most frequent words.

    Optional dependency ``wordfreq`` is used to obtain the most frequent terms
    for the chosen language. If any of the required libraries are missing,
    an :class:`ImportError` is raised.
    """
    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_bert_corpus")
    words = top_n_list(language, n_words)

    def get_vector(term):
        return term, get_word_vector_bert(term, model_name, output_layer)

    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_vector, words),
                                 total=len(words),
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
