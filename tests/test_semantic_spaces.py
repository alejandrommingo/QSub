import os
from pathlib import Path

import numpy as np

import QSub.semantic_spaces as spaces

def test_word_vector():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"
    test_space_dimensions = 300

    # Ejecutamos la función
    resultado = spaces.get_word_vector_gallito(test_word, test_code, test_space_name)

    # Comprobamos los tests
    assert isinstance(resultado, np.ndarray)
    assert resultado.size != 0
    assert resultado.shape == (test_space_dimensions,)

def test_word_cosine_similarity():
    # Parámetros de prueba
    test_word_a = "china"
    test_word_b = "corea_del_norte"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado_a = spaces.get_word_vector_gallito(test_word_a, test_code, test_space_name)
    resultado_b = spaces.get_word_vector_gallito(test_word_b, test_code, test_space_name)

    resultado = spaces.word_cosine_similarity(resultado_a, resultado_b)

    # Comprobamos los tests
    assert isinstance(resultado, np.float64)
    assert resultado is not None

def test_lsa_corpus():
    # Parámetros de prueba
    resources = Path(__file__).resolve().parent.parent / "resources"
    test_vocabulary_path = resources / "vocabulario_test_sp.txt"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado = spaces.get_lsa_corpus_gallito(str(test_vocabulary_path), test_code, test_space_name)

    # Comprobamos los tests
    assert isinstance(resultado, dict)
    assert isinstance(list(resultado.keys())[0], str)
    assert isinstance(list(resultado.values())[0], np.ndarray)


def test_word_cosine_similarity_zero_vector():
    """La similitud coseno con un vector nulo debe ser 0."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 0.0, 0.0])

    resultado = spaces.word_cosine_similarity(vec_a, vec_b)

    assert resultado == 0.0

def test_word_vector_bert_last():
    result = spaces.get_word_vector_bert("hello")
    assert isinstance(result, np.ndarray)
    assert result.shape == (768,)


def test_word_vector_bert_all_layers():
    result = spaces.get_word_vector_bert("hello", output_layer="all")
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 768

def test_bert_corpus():
    data = spaces.get_bert_corpus(language="en", n_words=5)
    assert isinstance(data, dict)
    assert len(data) == 5
    assert isinstance(list(data.values())[0], np.ndarray)


def test_bert_corpus_all_layers():
    data = spaces.get_bert_corpus(language="en", n_words=3, output_layer="all")
    assert isinstance(list(data.values())[0], np.ndarray)
    assert list(data.values())[0].shape[1] == 768


def test_word_vector_gpt2_last():
    result = spaces.get_word_vector_gpt2("hello")
    assert isinstance(result, np.ndarray)


def test_gpt2_corpus():
    data = spaces.get_gpt2_corpus(language="en", n_words=2)
    assert isinstance(data, dict)
    assert len(data) == 2


def test_word_vector_word2vec():
    result = spaces.get_word_vector_word2vec("hello")
    assert isinstance(result, np.ndarray)


def test_word2vec_corpus():
    data = spaces.get_word2vec_corpus(language="en", n_words=2)
    assert isinstance(data, dict)


def test_word_vector_glove():
    result = spaces.get_word_vector_glove("hello")
    assert isinstance(result, np.ndarray)


def test_glove_corpus():
    data = spaces.get_glove_corpus(language="en", n_words=2)
    assert isinstance(data, dict)


def test_word_vector_elmo_average():
    result = spaces.get_word_vector_elmo("hello")
    assert isinstance(result, np.ndarray)


def test_elmo_corpus():
    data = spaces.get_elmo_corpus(language="en", n_words=2)
    assert isinstance(data, dict)


def test_word_vector_distilbert():
    result = spaces.get_word_vector_distilbert("hello")
    assert isinstance(result, np.ndarray)


def test_distilbert_corpus():
    data = spaces.get_distilbert_corpus(language="en", n_words=2)
    assert isinstance(data, dict)

