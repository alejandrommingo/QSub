import os
import numpy as np
from QSub.semantic_spaces import get_word_vector_gallito, word_cosine_similarity, get_lsa_corpus_gallito

def test_word_vector():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_space_dimensions = 300

    # Ejecutamos la función
    resultado = get_word_vector_gallito(test_word, test_code, test_space_name)

    # Comprobamos los tests
    assert type(resultado) is np.ndarray  # Comprueba que el resultado es un array
    assert not resultado.size == 0  # Comprueba que el resultado no esta vacio
    assert resultado.shape == (test_space_dimensions,) # Comprueba que el shape de la matriz resultante es la esperada

def test_word_cosine_similarity():
    # Parámetros de prueba
    test_word_a = "china"
    test_word_b = "corea_del_norte"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado_a = get_word_vector_gallito(test_word_a, test_code, test_space_name)
    resultado_b = get_word_vector_gallito(test_word_b, test_code, test_space_name)

    resultado = word_cosine_similarity(resultado_a, resultado_b)

    # Comprobamos los tests
    assert type(resultado) is np.float64  # Comprueba que el resultado es un float
    assert resultado is not None  # Comprueba que el resultado no esta vacio

def test_lsa_corpus():
    # Parámetros de prueba
    test_vocabulary_path = "/home/alex/Documents/RESEARCH/QSub/resources/vocabulario_test_sp.txt" # Cambiar la ruta absoluta para tests en otros dispositivos
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado = get_lsa_corpus_gallito(test_vocabulary_path, test_code, test_space_name)

    # Comprobamos los tests
    assert type(resultado) == dict  # Comprueba que el resultado es un dict
    assert type(list(resultado.keys())[0]) == str  # Comprueba que las keys son strings
    assert type(list(resultado.values())[0]) == np.ndarray  # Comprueba que los values son ndarrays