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

def _detect_model_type(model_name: str) -> str:
    """
    Detecta automáticamente el tipo de modelo basado en el nombre.
    
    :param model_name: Nombre del modelo (ej: "bert-base-uncased", "gpt2", "distilbert-base-uncased")
    :return: Tipo de modelo ("bert", "gpt2", "distilbert", etc.)
    """
    model_name_lower = model_name.lower()
    
    # Detección por patrones comunes
    if any(pattern in model_name_lower for pattern in ["bert-", "roberta-", "albert-"]):
        return "bert"
    elif model_name_lower.startswith("distilbert"):
        return "bert"  # DistilBERT usa la misma arquitectura que BERT
    elif any(pattern in model_name_lower for pattern in ["gpt2", "gpt-2"]):
        return "gpt2"
    elif model_name_lower.startswith("gpt"):
        return "gpt2"  # GPT variants
    else:
        # Por defecto, asumimos BERT para modelos no reconocidos
        return "bert"

def _load_bert(model_name):
    """Load tokenizer and model for ``model_name`` with hidden states."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    return tokenizer, model

def _find_char_spans(text: str, term: str, case_sensitive: bool = False):
    """Devuelve lista de spans (start_char, end_char) de 'term' en 'text'."""
    haystack = text if case_sensitive else text.lower()
    needle   = term if case_sensitive else term.lower()

    spans = []
    start = 0
    while True:
        i = haystack.find(needle, start)
        if i == -1:
            break
        spans.append((i, i + len(needle)))
        start = i + len(needle)
    return spans


def get_static_word_vector(
    word: str,
    model_name: str = "bert-base-uncased",
    model_type: str = "auto",
    aggregation: str = "mean",  # "mean" | "first" | "sum" | "max"
):
    """
    Representación estática de `word` usando cualquier modelo transformer compatible.
    
    :param word: La palabra para la cual obtener el vector.
    :type word: str
    :param model_name: Nombre del modelo (ej: "bert-base-uncased", "gpt2", "distilbert-base-uncased").
    :type model_name: str
    :param model_type: Tipo de arquitectura ("auto", "bert", "gpt2"). Si es "auto", se detecta automáticamente.
    :type model_type: str
    :param aggregation: Método de agregación para subpalabras ("mean", "first", "sum", "max").
    :type aggregation: str
    :return: Vector numpy que representa la palabra.
    :rtype: numpy.ndarray
    """
    # Detectar tipo automáticamente si es necesario
    if model_type == "auto":
        model_type = _detect_model_type(model_name)
    
    # Delegar a la implementación específica según el tipo
    if model_type == "bert":
        return _get_static_word_vector_bert(word, model_name, aggregation)
    elif model_type == "gpt2":
        return _get_static_word_vector_gpt2(word, model_name, aggregation)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}. Use 'bert' o 'gpt2'.")


def _get_static_word_vector_bert(
    word: str,
    model_name: str = "bert-base-uncased",
    aggregation: str = "mean",
):
    """Implementación interna para vectores estáticos BERT."""
    if model_name not in _bert_models:
        _bert_models[model_name] = _load_bert(model_name)
    tokenizer, model = _bert_models[model_name]

    # 1) Tokeniza sin especiales para obtener SOLO subpiezas reales de la palabra
    token_ids = tokenizer.encode(word, add_special_tokens=False)

    if len(token_ids) == 0:
        raise ValueError(f"No se pudo tokenizar la palabra: {word!r}")

    # 2) Extrae la matriz de embeddings (vocab_size x hidden)
    emb_matrix = model.get_input_embeddings().weight  # no requiere grad

    # 3) Recolecta los embeddings de cada subpieza
    sub_embs = emb_matrix[token_ids, :]  # (n_subtokens, hidden)

    # 4) Agrega subpiezas → vector único
    if aggregation == "mean":
        vec = sub_embs.mean(dim=0)
    elif aggregation == "first":
        vec = sub_embs[0]
    elif aggregation == "sum":
        vec = sub_embs.sum(dim=0)
    elif aggregation == "max":
        vec, _ = sub_embs.max(dim=0)
    else:
        raise ValueError("aggregation debe ser 'mean' | 'first' | 'sum' | 'max'")

    return vec.detach().cpu().numpy()


def _get_static_word_vector_gpt2(
    word: str,
    model_name: str = "gpt2",
    aggregation: str = "mean",
):
    """Implementación interna para vectores estáticos GPT2."""
    if model_name not in _gpt2_models:
        _gpt2_models[model_name] = _load_gpt2(model_name)
    tokenizer, model = _gpt2_models[model_name]

    # 1) Tokeniza sin especiales para obtener SOLO subpiezas reales de la palabra
    token_ids = tokenizer.encode(word, add_special_tokens=False)

    if len(token_ids) == 0:
        raise ValueError(f"No se pudo tokenizar la palabra: {word!r}")

    # 2) Extrae la matriz de embeddings (vocab_size x hidden)
    emb_matrix = model.get_input_embeddings().weight  # no requiere grad

    # 3) Recolecta los embeddings de cada subpieza
    sub_embs = emb_matrix[token_ids, :]  # (n_subtokens, hidden)

    # 4) Agrega subpiezas → vector único
    if aggregation == "mean":
        vec = sub_embs.mean(dim=0)
    elif aggregation == "first":
        vec = sub_embs[0]
    elif aggregation == "sum":
        vec = sub_embs.sum(dim=0)
    elif aggregation == "max":
        vec, _ = sub_embs.max(dim=0)
    else:
        raise ValueError("aggregation debe ser 'mean' | 'first' | 'sum' | 'max'")

    return vec.detach().cpu().numpy()


def get_contextual_word_vector(
    term: str,
    text: str,
    model_name: str = "bert-base-uncased",
    model_type: str = "auto",
    output_layer="last",          # "last" | "all" | int (0..L-1)
    occurrence_index: int = 0,    # qué ocurrencia usar si hay varias
    aggregation: str = "mean",    # "mean" | "first" | "sum" | "max"
    case_sensitive: bool = False,
    complex_vector: bool = False, # Si True, devuelve vector complejo con embedding posicional
):
    """
    Representación contextual de una ocurrencia de `term` dentro de `text` usando cualquier modelo transformer.

    :param term: El término objetivo dentro del texto.
    :type term: str
    :param text: El texto completo que contiene el término.
    :type text: str
    :param model_name: Nombre del modelo a usar (ej: "bert-base-uncased", "gpt2").
    :type model_name: str
    :param model_type: Tipo de arquitectura ("auto", "bert", "gpt2"). Si es "auto", se detecta automáticamente.
    :type model_type: str
    :param output_layer: Capa del modelo a extraer. "last", "all", o índice entero.
    :type output_layer: str | int
    :param occurrence_index: Índice de la ocurrencia si hay múltiples apariciones.
    :type occurrence_index: int
    :param aggregation: Método de agregación para subpalabras.
    :type aggregation: str
    :param case_sensitive: Si la búsqueda del término es sensible a mayúsculas.
    :type case_sensitive: bool
    :param complex_vector: Si True, devuelve vector complejo con embeddings posicionales.
    :type complex_vector: bool
    :return: Vector numpy real o complejo según complex_vector.
    :rtype: numpy.ndarray
    """
    # Detectar tipo automáticamente si es necesario
    if model_type == "auto":
        model_type = _detect_model_type(model_name)
    
    # Delegar a la implementación específica según el tipo
    if model_type == "bert":
        return _get_contextual_word_vector_bert(
            term, text, model_name, output_layer, occurrence_index, 
            aggregation, case_sensitive, complex_vector
        )
    elif model_type == "gpt2":
        return _get_contextual_word_vector_gpt2(
            term, text, model_name, output_layer, occurrence_index,
            aggregation, case_sensitive, complex_vector
        )
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}. Use 'bert' o 'gpt2'.")


def _get_contextual_word_vector_bert(
    term: str,
    text: str,
    model_name: str = "bert-base-uncased",
    output_layer="last",
    occurrence_index: int = 0,
    aggregation: str = "mean",
    case_sensitive: bool = False,
    complex_vector: bool = False,
):
    """Implementación interna para vectores contextuales BERT."""
    if model_name not in _bert_models:
        _bert_models[model_name] = _load_bert(model_name)
    tokenizer, model = _bert_models[model_name]

    # 1) Encuentra spans por caracteres del término en el texto
    spans = _find_char_spans(text, term, case_sensitive=case_sensitive)
    if not spans:
        raise ValueError(f"No se encontró el término {term!r} en el texto.")
    if occurrence_index < 0 or occurrence_index >= len(spans):
        raise ValueError(f"occurrence_index fuera de rango (0..{len(spans)-1}).")

    target_span = spans[occurrence_index]  # (start_char, end_char)

    # 2) Tokeniza el párrafo con offsets para alinear char↔token
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,           # para textos largos; ajusta si quieres manejar sliding window
        max_length=512
    )
    offsets    = enc["offset_mapping"][0]   # (seq_len, 2)

    # 3) Filtra índices de tokens que cubren la ocurrencia (ignora especiales: offsets=(0,0))
    token_indices = []
    t_start, t_end = target_span
    for i, (s, e) in enumerate(offsets.tolist()):
        if s == e == 0:
            # típico de [CLS]/[SEP] en tokenizers rápidos
            continue
        # criterio de solape: cualquier intersección entre [s,e) y [t_start,t_end)
        if not (e <= t_start or s >= t_end):
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError("No se pudo mapear el término a tokens (revisa tokenización/offsets).")

    # 4) Pasa el texto por el modelo
    with torch.no_grad():
        outputs = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})

    # 5) Función auxiliar para agregar subtokens
    def _aggregate_subtokens(mat_tokens):  # (n_subtokens, hidden) -> (hidden,)
        if aggregation == "mean":
            return mat_tokens.mean(dim=0)
        elif aggregation == "first":
            return mat_tokens[0]
        elif aggregation == "sum":
            return mat_tokens.sum(dim=0)
        elif aggregation == "max":
            return mat_tokens.max(dim=0).values
        else:
            raise ValueError("aggregation debe ser 'mean' | 'first' | 'sum' | 'max'")

    # 6) Extraer embeddings posicionales si se necesitan vectores complejos
    def _get_positional_embeddings():
        """Extrae embeddings posicionales para los tokens del término."""
        if hasattr(model, 'embeddings') and hasattr(model.embeddings, 'position_embeddings'):
            # BERT tiene embeddings posicionales específicos
            pos_emb_matrix = model.embeddings.position_embeddings.weight
            # Tomar las posiciones correspondientes a nuestros tokens
            positions = torch.tensor(token_indices, dtype=torch.long)
            pos_embeddings = pos_emb_matrix[positions]  # (n_subtokens, hidden)
            return _aggregate_subtokens(pos_embeddings).detach().cpu().numpy()
        else:
            # Fallback: usar información de posición normalizada
            max_pos = len(offsets)
            avg_position = sum(token_indices) / len(token_indices) / max_pos
            hidden_size = outputs.last_hidden_state.shape[-1]
            # Crear patrón senoidal basado en la posición promedio
            pos_pattern = np.sin(np.linspace(0, 2*np.pi*avg_position, hidden_size))
            return pos_pattern

    # 7) Selecciona capa(s) y calcula resultado
    if output_layer == "last":
        # last_hidden_state: (1, seq_len, hidden)
        seq = outputs.last_hidden_state[0]                  # (seq_len, hidden)
        sub = seq[torch.tensor(token_indices, dtype=torch.long)]
        vec = _aggregate_subtokens(sub).cpu().numpy()
        
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            return vec + 1j * pos_vec
        return vec

    # hidden_states: tuple con embeddings y L capas
    hs = outputs.hidden_states
    if output_layer == "all":
        per_layer = []
        for h in hs[1:]:                                    # hs[0] = embeddings; 1..L = capas transformer
            seq = h[0]                                      # (seq_len, hidden)
            sub = seq[torch.tensor(token_indices, dtype=torch.long)]
            layer_vec = _aggregate_subtokens(sub).cpu().numpy()
            per_layer.append(layer_vec)
        
        result = np.stack(per_layer)                        # (L, hidden)
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            # Repetir el vector posicional para todas las capas
            pos_matrix = np.broadcast_to(pos_vec, result.shape)
            return result + 1j * pos_matrix
        return result

    if isinstance(output_layer, int):
        num_layers = len(hs) - 1
        if output_layer < 0 or output_layer >= num_layers:
            raise ValueError(f"output_layer fuera de rango (0..{num_layers-1})")
        seq = hs[1 + output_layer][0]                       # (seq_len, hidden)
        sub = seq[torch.tensor(token_indices, dtype=torch.long)]
        vec = _aggregate_subtokens(sub).cpu().numpy()
        
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            return vec + 1j * pos_vec
        return vec

    raise ValueError("output_layer debe ser 'last', 'all' o un entero 0..L-1")


def _get_contextual_word_vector_gpt2(
    term: str,
    text: str,
    model_name: str = "gpt2",
    output_layer="last",
    occurrence_index: int = 0,
    aggregation: str = "mean",
    case_sensitive: bool = False,
    complex_vector: bool = False,
):
    """Implementación interna para vectores contextuales GPT2."""
    if model_name not in _gpt2_models:
        _gpt2_models[model_name] = _load_gpt2(model_name)
    tokenizer, model = _gpt2_models[model_name]

    # 1) Encuentra spans por caracteres del término en el texto
    spans = _find_char_spans(text, term, case_sensitive=case_sensitive)
    if not spans:
        raise ValueError(f"No se encontró el término {term!r} en el texto.")
    if occurrence_index < 0 or occurrence_index >= len(spans):
        raise ValueError(f"occurrence_index fuera de rango (0..{len(spans)-1}).")

    target_span = spans[occurrence_index]  # (start_char, end_char)

    # 2) Tokeniza el texto con offsets para alinear char↔token
    enc = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=1024  # GPT2 tiene contexto más largo que BERT
    )
    offsets    = enc["offset_mapping"][0]   # (seq_len, 2)

    # 3) Filtra índices de tokens que cubren la ocurrencia
    token_indices = []
    t_start, t_end = target_span
    for i, (s, e) in enumerate(offsets.tolist()):
        if s == e == 0:
            # tokens especiales
            continue
        # criterio de solape: cualquier intersección entre [s,e) y [t_start,t_end)
        if not (e <= t_start or s >= t_end):
            token_indices.append(i)

    if not token_indices:
        raise RuntimeError("No se pudo mapear el término a tokens (revisa tokenización/offsets).")

    # 4) Pasa el texto por el modelo
    with torch.no_grad():
        outputs = model(**{k: v for k, v in enc.items() if k != "offset_mapping"})

    # 5) Función auxiliar para agregar subtokens
    def _aggregate_subtokens(mat_tokens):  # (n_subtokens, hidden) -> (hidden,)
        if aggregation == "mean":
            return mat_tokens.mean(dim=0)
        elif aggregation == "first":
            return mat_tokens[0]
        elif aggregation == "sum":
            return mat_tokens.sum(dim=0)
        elif aggregation == "max":
            return mat_tokens.max(dim=0).values
        else:
            raise ValueError("aggregation debe ser 'mean' | 'first' | 'sum' | 'max'")

    # 6) Extraer embeddings posicionales si se necesitan vectores complejos
    def _get_positional_embeddings():
        """Extrae embeddings posicionales para los tokens del término."""
        if hasattr(model, 'wpe'):  # GPT2 usa 'wpe' para position embeddings
            pos_emb_matrix = model.wpe.weight
            # Tomar las posiciones correspondientes a nuestros tokens
            positions = torch.tensor(token_indices, dtype=torch.long)
            if positions.max() < pos_emb_matrix.size(0):
                pos_embeddings = pos_emb_matrix[positions]  # (n_subtokens, hidden)
                return _aggregate_subtokens(pos_embeddings).detach().cpu().numpy()
        
        # Fallback: usar información de posición normalizada
        max_pos = len(offsets)
        avg_position = sum(token_indices) / len(token_indices) / max_pos
        hidden_size = outputs.last_hidden_state.shape[-1]
        # Crear patrón senoidal basado en la posición promedio
        pos_pattern = np.sin(np.linspace(0, 2*np.pi*avg_position, hidden_size))
        return pos_pattern

    # 7) Selecciona capa(s) y calcula resultado
    if output_layer == "last":
        # last_hidden_state: (1, seq_len, hidden)
        seq = outputs.last_hidden_state[0]                  # (seq_len, hidden)
        sub = seq[torch.tensor(token_indices, dtype=torch.long)]
        vec = _aggregate_subtokens(sub).cpu().numpy()
        
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            return vec + 1j * pos_vec
        return vec

    # hidden_states: tuple con embeddings y L capas
    hs = outputs.hidden_states
    if output_layer == "all":
        per_layer = []
        for h in hs[1:]:                                    # hs[0] = embeddings; 1..L = capas transformer
            seq = h[0]                                      # (seq_len, hidden)
            sub = seq[torch.tensor(token_indices, dtype=torch.long)]
            layer_vec = _aggregate_subtokens(sub).cpu().numpy()
            per_layer.append(layer_vec)
        
        result = np.stack(per_layer)                        # (L, hidden)
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            # Repetir el vector posicional para todas las capas
            pos_matrix = np.broadcast_to(pos_vec, result.shape)
            return result + 1j * pos_matrix
        return result

    if isinstance(output_layer, int):
        num_layers = len(hs) - 1
        if output_layer < 0 or output_layer >= num_layers:
            raise ValueError(f"output_layer fuera de rango (0..{num_layers-1})")
        seq = hs[1 + output_layer][0]                       # (seq_len, hidden)
        sub = seq[torch.tensor(token_indices, dtype=torch.long)]
        vec = _aggregate_subtokens(sub).cpu().numpy()
        
        if complex_vector:
            pos_vec = _get_positional_embeddings()
            return vec + 1j * pos_vec
        return vec

    raise ValueError("output_layer debe ser 'last', 'all' o un entero 0..L-1")


# Nota: API unificada implementada. Las funciones principales son:
# - get_static_word_vector(word, model_name, model_type="auto") para representaciones estáticas
# - get_contextual_word_vector(term, text, model_name, model_type="auto") para representaciones contextuales
# Soporta detección automática de tipos de modelo o especificación manual.


def get_bert_corpus(
    language="en",
    model_name="bert-base-uncased",
    n_words=1000,
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
        return term, get_static_word_vector(term, model_name, model_type="bert")

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


# Nota: API unificada implementada. Las funciones principales son:
# - get_static_word_vector(word, model_name, model_type="auto") para representaciones estáticas
# - get_contextual_word_vector(term, text, model_name, model_type="auto") para representaciones contextuales
# Soporta detección automática de tipos de modelo o especificación manual.


def get_gpt2_corpus(
    language="en",
    model_name="gpt2",
    n_words=1000,
):
    """Return a dictionary with GPT2 vectors of the most frequent words."""

    if top_n_list is None:
        raise ImportError("wordfreq must be installed to use get_gpt2_corpus")
    words = top_n_list(language, n_words)

    def get_vector(term):
        return term, get_static_word_vector(term, model_name, model_type="gpt2")

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

def get_word_vector_distilbert(word, model_name="distilbert-base-uncased"):
    """Wrapper around get_static_word_vector for DistilBERT."""
    return get_static_word_vector(word, model_name=model_name, model_type="bert")


def get_distilbert_corpus(
    language="en",
    model_name="distilbert-base-uncased",
    n_words=1000,
):
    """Wrapper around get_bert_corpus for DistilBERT."""
    return get_bert_corpus(language, model_name=model_name, n_words=n_words)

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
