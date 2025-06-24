import os
from pathlib import Path

import numpy as np
import pandas as pd

import QSub.contours as contours
import QSub.semantic_spaces as spaces

def test_gallito_neighbors_matrix():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    resultado = contours.get_neighbors_matrix_gallito(
        test_word, test_code, test_space_name, neighbors=test_neighbors
    )

    assert isinstance(resultado, dict)
    assert isinstance(list(resultado.keys())[0], str)
    assert isinstance(list(resultado.values())[0], np.ndarray)

def test_neighbors_similarity():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    neighbors = contours.get_neighbors_matrix_gallito(
        test_word, test_code, test_space_name, neighbors=test_neighbors
    )
    word = spaces.get_word_vector_gallito(test_word, test_code, test_space_name)
    resultado = contours.neighbors_similarity(word, neighbors)

    assert isinstance(resultado, dict)
    assert isinstance(list(resultado.keys())[0], str)
    assert isinstance(list(resultado.values())[0], np.float64)

def test_get_superterm():
    # Parámetros de prueba
    resources = Path(__file__).resolve().parent.parent / "resources"
    test_vocabulary_path = resources / "vocabulario_test_sp.txt"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado = contours.get_superterm_gallito(str(test_vocabulary_path), test_code, test_space_name)

    # Comprobamos los tests
    assert len(resultado) == 2
    assert isinstance(resultado, tuple)
    assert isinstance(resultado[0], np.ndarray)
    assert isinstance(resultado[1], np.ndarray)

def test_deserved_neighbors():
    resources = Path(__file__).resolve().parent.parent / "resources"
    test_h_df_path = resources / "sp_vocab_semantic_diversity.csv"
    test_superterm_cosines_path = resources / "superterm_cosines.csv"
    test_word = "china"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 100

    test_h_df = pd.read_csv(test_h_df_path)
    test_h_df = test_h_df.drop(test_h_df.columns[0], axis=1)
    test_superterm_cosines = pd.read_csv(test_superterm_cosines_path)
    test_superterm_cosines = test_superterm_cosines.drop(test_superterm_cosines.columns[0], axis=1)
    test_superterm_cosines = np.array(test_superterm_cosines)

    # Ejecutamos la función
    neighbors = contours.get_neighbors_matrix_gallito(
        test_word, test_code, test_space_name, neighbors=test_neighbors
    )
    word = spaces.get_word_vector_gallito(test_word, test_code, test_space_name)
    word_cosines = contours.neighbors_similarity(word, neighbors)
    word_cosines = np.array(list(word_cosines.values()))
    resultado = contours.deserved_neighbors(
        "chino", test_h_df, test_superterm_cosines, word_cosines
    )

    # Comprobamos los tests
    assert resultado > 0
    assert isinstance(resultado, int)

def test_find_closest_neighbors_lsa():
    # Parámetros de prueba
    resources = Path(__file__).resolve().parent.parent / "resources"
    test_word = "mundo"
    test_vocabulary_path = resources / "vocabulario_test_sp.txt"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 5

    # Ejecutamos la función
    test_corpus = spaces.get_lsa_corpus_gallito(str(test_vocabulary_path), test_code, test_space_name)
    resultado = contours.find_closest_neighbors_lsa(test_word, test_corpus, n_neighbors=test_neighbors)

    # Comprobamos los tests
    assert isinstance(resultado, dict)
    assert isinstance(list(resultado.keys())[0], str)
    assert isinstance(list(resultado.values())[0], np.ndarray)


def test_get_neighbors_matrix_bert():
    result = contours.get_neighbors_matrix_bert("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_neighbors_matrix_gpt2():
    result = contours.get_neighbors_matrix_gpt2("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_neighbors_matrix_word2vec():
    result = contours.get_neighbors_matrix_word2vec("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_neighbors_matrix_glove():
    result = contours.get_neighbors_matrix_glove("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_neighbors_matrix_elmo():
    result = contours.get_neighbors_matrix_elmo("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_neighbors_matrix_distilbert():
    result = contours.get_neighbors_matrix_distilbert("hello")
    assert isinstance(result, dict)
    assert isinstance(list(result.values())[0], np.ndarray)


def test_get_contextual_contour_wikipedia():
    result = contours.get_contextual_contour_wikipedia("hello")
    assert isinstance(result, dict)
    if result:
        assert isinstance(list(result.values())[0], np.ndarray)

