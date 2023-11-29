import os
import numpy as np
from QSub.contours import gallito_neighbors_matrix, word_vector, word_cosine_similarity, neighbors_similarity


def test_gallito_neighbors_matrix():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10
    test_space_dimensions = 300

    # Ejecutamos la función
    resultado = gallito_neighbors_matrix(test_word, test_code, test_space_name, neighbors=test_neighbors, space_dimensions=test_space_dimensions)

    # Comprobamos los tests
    assert type(resultado["neighbors_vec"]) is np.ndarray  # Comprueba que el resultado es una matriz
    assert not resultado["neighbors_vec"].size == 0  # Comprueba que el resultado no esta vacio
    assert resultado["neighbors_vec"].shape == (test_neighbors, test_space_dimensions) # Comprueba que el shape de la matriz resultante es la esperada

def test_word_vector():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_space_dimensions = 300

    # Ejecutamos la función
    resultado = word_vector(test_word, test_code, test_space_name)

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
    resultado_a = word_vector(test_word_a, test_code, test_space_name)
    resultado_b = word_vector(test_word_b, test_code, test_space_name)

    resultado = word_cosine_similarity(resultado_a, resultado_b)

    # Comprobamos los tests
    assert type(resultado) is np.float64  # Comprueba que el resultado es un float
    assert resultado is not None  # Comprueba que el resultado no esta vacio

def test_neighbors_similarity():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10
    test_space_dimensions = 300

    # Ejecutamos la función
    neighbors = gallito_neighbors_matrix(test_word, test_code, test_space_name, neighbors=test_neighbors, space_dimensions=test_space_dimensions)
    word = word_vector(test_word, test_code, test_space_name)
    resultado = neighbors_similarity(word, neighbors)

    # Comprobamos los tests
    assert len(resultado) == 2  # Comprueba que el resultado tenga dos elementos
    assert type(resultado) == tuple  # Comprueba que el resultado es una tupla
    assert type(resultado[0]) == np.ndarray # Comprueba que los elementos de la tupla son ndarrays
    assert type(resultado[1]) == np.ndarray