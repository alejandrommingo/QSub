import os

import numpy as np

from QSub.contours import gallito_neighbors_matrix, word_vector


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
    assert type(resultado) is np.ndarray  # Comprueba que el resultado es una matriz
    assert not resultado.size == 0  # Comprueba que el resultado no esta vacio
    assert resultado.shape == (test_neighbors, test_space_dimensions) # Comprueba que el shape de la matriz resultante es la esperada

def test_word_vector():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_space_dimensions = 300

    # Ejecutamos la función
    resultado = word_vector(test_word, test_code, test_space_name)

    # Comprobamos los tests
    assert type(resultado) is np.ndarray  # Comprueba que el resultado es una matriz
    assert not resultado.size == 0  # Comprueba que el resultado no esta vacio
    assert resultado.shape == (test_space_dimensions,) # Comprueba que el shape de la matriz resultante es la esperada