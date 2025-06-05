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

try:  # pragma: no cover - optional dependency
    import gensim.downloader as gensim_api
except ImportError:  # pragma: no cover - optional dependency
    gensim_api = None

try:  # pragma: no cover - optional dependency
    from allennlp.commands.elmo import ElmoEmbedder
except ImportError:  # pragma: no cover - optional dependency
    ElmoEmbedder = None

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
_gpt2_models = {}
_w2v_models = {}
_glove_models = {}
_elmo_models = {}

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

#############################################
## GPT2 BASED SEMANTIC SPACE OPERATIONS    ##
#############################################

def _load_gpt2(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return tokenizer, model


def get_word_vector_gpt2(word, model_name="gpt2", output_layer="last"):
    """Return GPT2 embeddings for a given word.

    Parameters are analogous to :func:`get_word_vector_bert`.
    """

    if AutoTokenizer is None or AutoModel is None or torch is None:
        raise ImportError(
            "transformers and torch must be installed to use get_word_vector_gpt2"
        )

    if model_name not in _gpt2_models:
        _gpt2_models[model_name] = _load_gpt2(model_name)
    tokenizer, model = _gpt2_models[model_name]
    inputs = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    if output_layer == "last":
        vector = outputs.last_hidden_state.mean(dim=1).squeeze()
        return vector.numpy()

    if output_layer == "all":
        hidden = [h.mean(dim=1).squeeze().numpy() for h in outputs.hidden_states]
        return np.stack(hidden)

    if isinstance(output_layer, int):
        hidden_states = outputs.hidden_states
        if output_layer < 0 or output_layer >= len(hidden_states):
            raise ValueError("output_layer out of range")
        vector = hidden_states[output_layer].mean(dim=1).squeeze()
        return vector.numpy()

    raise ValueError("output_layer must be 'last', 'all', or an integer index")


def get_gpt2_corpus(
    language="en",
    model_name="gpt2",
    n_words=1000,
    output_layer="last",
):
    """Return a dictionary with GPT2 vectors of the most frequent words."""

    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_gpt2_corpus")
    words = top_n_list(language, n_words)

    def get_vector(term):
        return term, get_word_vector_gpt2(term, model_name, output_layer)

    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_vector, words),
                                 total=len(words),
                                 desc="Procesando términos"):
            corpus_dict[term] = np.array(vector)

    return corpus_dict

#############################################
## WORD2VEC AND GLOVE OPERATIONS          ##
#############################################

def _load_word2vec(model_name):
    return gensim_api.load(model_name)


def get_word_vector_word2vec(word, model_name="word2vec-google-news-300"):
    """Return Word2Vec embeddings for ``word`` from ``model_name``."""

    if gensim_api is None:
        raise ImportError("gensim must be installed to use get_word_vector_word2vec")

    if model_name not in _w2v_models:
        _w2v_models[model_name] = _load_word2vec(model_name)
    model = _w2v_models[model_name]
    return model[word]


def get_word2vec_corpus(
    language="en",
    model_name="word2vec-google-news-300",
    n_words=1000,
):
    """Return a dictionary with Word2Vec vectors of the most frequent words."""

    if gensim_api is None:
        raise ImportError("gensim must be installed to use get_word2vec_corpus")
    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_word2vec_corpus")

    words = top_n_list(language, n_words)

    def get_vector(term):
        try:
            return term, get_word_vector_word2vec(term, model_name)
        except KeyError:
            return term, None

    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_vector, words),
                                 total=len(words),
                                 desc="Procesando términos"):
            if vector is not None:
                corpus_dict[term] = np.array(vector)

    return corpus_dict


def _load_glove(model_name):
    return gensim_api.load(model_name)


def get_word_vector_glove(word, model_name="glove-wiki-gigaword-300"):
    """Return GloVe embeddings for ``word`` from ``model_name``."""

    if gensim_api is None:
        raise ImportError("gensim must be installed to use get_word_vector_glove")

    if model_name not in _glove_models:
        _glove_models[model_name] = _load_glove(model_name)
    model = _glove_models[model_name]
    return model[word]


def get_glove_corpus(
    language="en",
    model_name="glove-wiki-gigaword-300",
    n_words=1000,
):
    """Return a dictionary with GloVe vectors of the most frequent words."""

    if gensim_api is None:
        raise ImportError("gensim must be installed to use get_glove_corpus")
    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_glove_corpus")

    words = top_n_list(language, n_words)

    def get_vector(term):
        try:
            return term, get_word_vector_glove(term, model_name)
        except KeyError:
            return term, None

    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_vector, words),
                                 total=len(words),
                                 desc="Procesando términos"):
            if vector is not None:
                corpus_dict[term] = np.array(vector)

    return corpus_dict

#############################################
## ELMO BASED SEMANTIC SPACE OPERATIONS   ##
#############################################

def _load_elmo(model_name="small"):
    return ElmoEmbedder(model_name=model_name)


def get_word_vector_elmo(word, model_name="small", output_layer="average"):
    """Return ELMo embeddings for a given word.

    ``output_layer`` can be ``"average"`` (default), ``"all"`` or the index of a
    specific layer.
    """

    if ElmoEmbedder is None:
        raise ImportError("allennlp must be installed to use get_word_vector_elmo")

    if model_name not in _elmo_models:
        _elmo_models[model_name] = _load_elmo(model_name)
    embedder = _elmo_models[model_name]
    vectors = embedder.embed_sentence([word])  # (layers, 1, dim)

    if output_layer == "average":
        return vectors.mean(axis=0).squeeze()

    if output_layer == "all":
        return vectors.squeeze()

    if isinstance(output_layer, int):
        if output_layer < 0 or output_layer >= vectors.shape[0]:
            raise ValueError("output_layer out of range")
        return vectors[output_layer, 0, :]

    raise ValueError("output_layer must be 'average', 'all', or an integer index")


def get_elmo_corpus(
    language="en",
    model_name="small",
    n_words=1000,
    output_layer="average",
):
    """Return a dictionary with ELMo vectors of the most frequent words."""

    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_elmo_corpus")
    words = top_n_list(language, n_words)

    def get_vector(term):
        return term, get_word_vector_elmo(term, model_name, output_layer)

    corpus_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for term, vector in tqdm(executor.map(get_vector, words),
                                 total=len(words),
                                 desc="Procesando términos"):
            corpus_dict[term] = np.array(vector)

    return corpus_dict

#############################################
## DISTILBERT BASED SEMANTIC SPACE OPS    ##
#############################################

def get_word_vector_distilbert(word, model_name="distilbert-base-uncased", output_layer="last"):
    """Wrapper around :func:`get_word_vector_bert` for DistilBERT."""
    return get_word_vector_bert(word, model_name=model_name, output_layer=output_layer)


def get_distilbert_corpus(
    language="en",
    model_name="distilbert-base-uncased",
    n_words=1000,
    output_layer="last",
):
    """Wrapper around :func:`get_bert_corpus` for DistilBERT."""
    return get_bert_corpus(language, model_name=model_name, n_words=n_words, output_layer=output_layer)

#######################################
## GENERAL SEMANTIC SPACE OPERATIONS ##
#######################################

def vector_add(v1, v2):
    """
    Suma dos vectores elemento a elemento.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Vector resultante de la suma ``v1 + v2``.
    :rtype: numpy.ndarray
    """
    return np.add(v1, v2)


def vector_subtract(v1, v2):
    """
    Resta dos vectores elemento a elemento.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Vector resultante de ``v1 - v2``.
    :rtype: numpy.ndarray
    """
    return np.subtract(v1, v2)


def scalar_multiply(scalar, v):
    """
    Multiplica un vector por un escalar.

    :param scalar: Valor escalar a multiplicar.
    :type scalar: float | int
    :param v: Vector objetivo.
    :type v: numpy.ndarray
    :return: Vector ``v`` escalado por ``scalar``.
    :rtype: numpy.ndarray
    """
    return scalar * v


def dot_product(v1, v2):
    """
    Calcula el producto punto entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Resultado escalar del producto ``v1 · v2``.
    :rtype: float
    """
    return float(np.dot(v1, v2))


def l2_norm(v):
    """
    Devuelve la norma euclidiana de un vector.

    :param v: Vector objetivo.
    :type v: numpy.ndarray
    :return: Valor de la norma L2 de ``v``.
    :rtype: float
    """
    return float(np.linalg.norm(v))


def l1_norm(v):
    """
    Devuelve la norma L1 de un vector.

    :param v: Vector objetivo.
    :type v: numpy.ndarray
    :return: Valor de la norma L1 de ``v``.
    :rtype: float
    """
    return float(np.linalg.norm(v, ord=1))


def linf_norm(v):
    """
    Devuelve la norma L∞ de un vector.

    :param v: Vector objetivo.
    :type v: numpy.ndarray
    :return: Valor de la norma L∞ de ``v``.
    :rtype: float
    """
    return float(np.linalg.norm(v, ord=np.inf))


def unit_vector(v):
    """
    Normaliza un vector para que tenga norma unitaria.

    :param v: Vector a normalizar.
    :type v: numpy.ndarray
    :return: Vector ``v`` normalizado.
    :rtype: numpy.ndarray
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def zscore(v):
    """
    Estandariza un vector restando la media y dividiendo por la desviación típica.

    :param v: Vector a estandarizar.
    :type v: numpy.ndarray
    :return: Vector ``v`` transformado a puntuaciones z.
    :rtype: numpy.ndarray
    """
    mean = np.mean(v)
    std = np.std(v)
    if std == 0:
        return np.zeros_like(v)
    return (v - mean) / std


def covariance_matrix(X):
    """
    Calcula la matriz de covarianza de un conjunto de datos.

    :param X: Matriz de datos donde cada fila es una observación.
    :type X: numpy.ndarray
    :return: Matriz de covarianza de ``X``.
    :rtype: numpy.ndarray
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X_centered = X - np.mean(X, axis=0)
    n_samples = X_centered.shape[0]
    if n_samples <= 1:
        return np.zeros((X_centered.shape[1], X_centered.shape[1]))
    return np.dot(X_centered.T, X_centered) / (n_samples - 1)


def compute_svd(X):
    """
    Devuelve la descomposición en valores singulares (SVD) de una matriz.

    :param X: Matriz de entrada.
    :type X: numpy.ndarray
    :return: Tupla ``(U, S, Vt)`` resultante de la SVD.
    :rtype: tuple
    """
    return np.linalg.svd(X, full_matrices=False)

def cosine_similarity(v1, v2):
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




def euclidean_distance(v1, v2):
    """
    Devuelve la distancia euclídea entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Distancia euclídea entre ``v1`` y ``v2``.
    :rtype: float
    """
    return float(np.linalg.norm(v1 - v2))


def manhattan_distance(v1, v2):
    """
    Devuelve la distancia Manhattan (L1) entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Distancia L1 entre ``v1`` y ``v2``.
    :rtype: float
    """
    return float(np.linalg.norm(v1 - v2, ord=1))


def minkowski_distance(v1, v2, p=2):
    """
    Distancia de Minkowski generalizada entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :param p: Orden de la distancia.
    :type p: int | float
    :return: Distancia de Minkowski de orden ``p``.
    :rtype: float
    """
    return float(np.linalg.norm(v1 - v2, ord=p))


def chebyshev_distance(v1, v2):
    """
    Distancia de Chebyshev (L∞) entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Distancia L∞ entre ``v1`` y ``v2``.
    :rtype: float
    """
    return float(np.linalg.norm(v1 - v2, ord=np.inf))


def mahalanobis_distance(v1, v2, cov):
    """
    Calcula la distancia de Mahalanobis entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :param cov: Matriz de covarianza de los datos.
    :type cov: numpy.ndarray
    :return: Distancia de Mahalanobis.
    :rtype: float
    """
    diff = v1 - v2
    inv_cov = np.linalg.pinv(cov)
    return float(np.sqrt(diff.T @ inv_cov @ diff))


def weighted_minkowski_distance(v1, v2, p=2, w=None):
    """
    Distancia de Minkowski ponderada entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :param p: Orden de la distancia.
    :type p: int | float
    :param w: Pesos de cada dimensión.
    :type w: numpy.ndarray | None
    :return: Distancia ponderada de Minkowski.
    :rtype: float
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    if w is None:
        w = np.ones_like(v1)
    diff = np.abs(v1 - v2)
    return float(np.power(np.sum(w * diff ** p), 1.0 / p))


def hellinger_distance(p, q):
    """
    Distancia de Hellinger para distribuciones de probabilidad.

    :param p: Distribución de probabilidad ``p``.
    :type p: numpy.ndarray
    :param q: Distribución de probabilidad ``q``.
    :type q: numpy.ndarray
    :return: Distancia de Hellinger entre ``p`` y ``q``.
    :rtype: float
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2))


def kl_divergence(p, q, epsilon=1e-12):
    """
    Divergencia de Kullback-Leibler entre dos distribuciones.

    :param p: Distribución de probabilidad ``p``.
    :type p: numpy.ndarray
    :param q: Distribución de probabilidad ``q``.
    :type q: numpy.ndarray
    :param epsilon: Valor de suavizado para evitar logaritmos de cero.
    :type epsilon: float
    :return: Valor de la divergencia KL ``D(p‖q)``.
    :rtype: float
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, epsilon, None)
    q = np.clip(q, epsilon, None)
    return float(np.sum(p * np.log(p / q)))


def pearson_similarity(v1, v2):
    """
    Calcula la similitud de correlación de Pearson.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Coeficiente de correlación de Pearson entre ``v1`` y ``v2``.
    :rtype: float
    """
    v1 = np.asarray(v1)
    v2 = np.asarray(v2)
    v1_mean = v1 - np.mean(v1)
    v2_mean = v2 - np.mean(v2)
    numerator = np.dot(v1_mean, v2_mean)
    denominator = np.linalg.norm(v1_mean) * np.linalg.norm(v2_mean)
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def angular_distance(v1, v2):
    """
    Devuelve la distancia angular normalizada entre dos vectores.

    :param v1: Primer vector.
    :type v1: numpy.ndarray
    :param v2: Segundo vector.
    :type v2: numpy.ndarray
    :return: Distancia angular en el rango ``[0, 1]``.
    :rtype: float
    """
    cos_sim = cosine_similarity(v1, v2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    angle = np.arccos(cos_sim)
    return float(angle / np.pi)


def jaccard_similarity(v1, v2):
    """
    Calcula la similitud de Jaccard para vectores binarios.

    :param v1: Primer vector binario.
    :type v1: numpy.ndarray
    :param v2: Segundo vector binario.
    :type v2: numpy.ndarray
    :return: Coeficiente de Jaccard entre ``v1`` y ``v2``.
    :rtype: float
    """
    v1 = np.asarray(v1).astype(bool)
    v2 = np.asarray(v2).astype(bool)
    intersection = np.sum(np.logical_and(v1, v2))
    union = np.sum(np.logical_or(v1, v2))
    if union == 0:
        return 0.0
    return float(intersection / union)
