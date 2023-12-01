import os
import numpy as np
import pandas as pd
from QSub.contours import get_neighbors_matrix_gallito, neighbors_similarity, get_superterm_gallito, deserved_neighbors
from QSub.semantic_spaces import get_word_vector_gallito

def test_gallito_neighbors_matrix():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    # Ejecutamos la función
    resultado = get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)

    # Comprobamos los tests
    assert type(resultado) is dict  # Comprueba que el resultado es un dict
    assert type(list(resultado.keys())[0]) == str  # Comprueba que las keys son strings
    assert type(list(resultado.values())[0]) == np.ndarray # Comprueba que los values son ndarrays

def test_neighbors_similarity():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    # Ejecutamos la función
    neighbors = get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)
    word = get_word_vector_gallito(test_word, test_code, test_space_name)
    resultado = neighbors_similarity(word, neighbors)

    # Comprobamos los tests
    assert type(resultado) is dict  # Comprueba que el resultado es un dict
    assert type(list(resultado.keys())[0]) == str  # Comprueba que las keys son strings
    assert type(list(resultado.values())[0]) == np.float64  # Comprueba que los values son ndarrays

def test_get_superterm():
    # Parámetros de prueba
    test_vocabulary_path = "/home/alex/Documents/RESEARCH/QSub/resources/vocabulario_test_sp.txt" # Cambiar la ruta absoluta para tests en otros dispositivos
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado = get_superterm_gallito(test_vocabulary_path, test_code, test_space_name)

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

    # Importaciones de prueba
    test_h_df = pd.read_csv(test_h_df_path)  # Asegúrate de usar la ruta correcta al archivo
    test_h_df = test_h_df.drop(test_h_df.columns[0], axis=1)
    test_superterm_cosines = pd.read_csv(test_superterm_cosines_path)
    test_superterm_cosines = test_superterm_cosines.drop(test_superterm_cosines.columns[0], axis=1)
    test_superterm_cosines = np.array(test_superterm_cosines)

    # Ejecutamos la función
    neighbors = get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)
    word = get_word_vector_gallito(test_word, test_code, test_space_name)
    word_cosines = neighbors_similarity(word, neighbors)
    word_cosines = np.array(list(word_cosines.values()))
    resultado = deserved_neighbors("chino", test_h_df, test_superterm_cosines, word_cosines)

    # Comprobamos los tests
    assert resultado > 0  # Comprueba que el resultado es mayor que 0
    assert type(resultado) == int  # Comprueba que el resultado es un int