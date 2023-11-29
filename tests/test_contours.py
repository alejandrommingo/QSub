import os
import numpy as np
import pandas as pd
from QSub.contours import gallito_neighbors_matrix, word_vector, word_cosine_similarity, neighbors_similarity, get_superterm, deserved_neighbors


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

def test_get_superterm():
    # Parámetros de prueba
    test_vocabulary_path = "/home/alex/Documents/RESEARCH/QSub/resources/vocabulario_test_sp.txt" # Cambiar la ruta absoluta para tests en otros dispositivos
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado = get_superterm(test_vocabulary_path, test_code, test_space_name)

    # Comprobamos los tests
    assert len(resultado) == 2  # Comprueba que el resultado tenga dos elementos
    assert type(resultado) == tuple  # Comprueba que el resultado es una tupla
    assert type(resultado[0]) == np.ndarray # Comprueba que los elementos de la tupla son ndarrays
    assert type(resultado[1]) == np.ndarray

def test_deserved_neighbors():
    # Parámetros de prueba
    test_h_df_path = "/home/alex/Documents/RESEARCH/QSub/resources/sp_vocab_semantic_diversity.csv" # Cambiar la ruta absoluta para tests en otros dispositivos
    test_superterm_cosines_path = "/home/alex/Documents/RESEARCH/QSub/resources/superterm_cosines.csv"  # Cambiar la ruta absoluta para tests en otros dispositivos
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 100
    test_space_dimensions = 300

    # Importaciones de prueba
    test_h_df = pd.read_csv(test_h_df_path)  # Asegúrate de usar la ruta correcta al archivo
    test_h_df = test_h_df.drop(test_h_df.columns[0], axis=1)
    test_superterm_cosines = pd.read_csv(test_superterm_cosines_path)
    test_superterm_cosines = test_superterm_cosines.drop(test_superterm_cosines.columns[0], axis=1)
    test_superterm_cosines = np.array(test_superterm_cosines)

    # Ejecutamos la función
    neighbors = gallito_neighbors_matrix(test_word, test_code, test_space_name, neighbors=test_neighbors,
                                         space_dimensions=test_space_dimensions)
    word = word_vector(test_word, test_code, test_space_name)
    word_cosines = neighbors_similarity(word, neighbors)
    word_cosines = word_cosines[0]
    resultado = deserved_neighbors("chino", test_h_df, test_superterm_cosines, word_cosines)

    # Comprobamos los tests
    assert resultado > 0  # Comprueba que el resultado es mayor que 0
    assert type(resultado) == int  # Comprueba que el resultado es un int