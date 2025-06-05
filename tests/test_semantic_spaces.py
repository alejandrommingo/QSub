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

def test_cosine_similarity():
    # Parámetros de prueba
    test_word_a = "china"
    test_word_b = "corea_del_norte"
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    test_space_name = "quantumlikespace_spanish"

    # Ejecutamos la función
    resultado_a = spaces.get_word_vector_gallito(test_word_a, test_code, test_space_name)
    resultado_b = spaces.get_word_vector_gallito(test_word_b, test_code, test_space_name)

    resultado = spaces.cosine_similarity(resultado_a, resultado_b)

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


def test_cosine_similarity_zero_vector():
    """La similitud coseno con un vector nulo debe ser 0."""
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 0.0, 0.0])

    resultado = spaces.cosine_similarity(vec_a, vec_b)

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


def test_vector_add_subtract():
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    assert np.array_equal(spaces.vector_add(v1, v2), np.array([5, 7, 9]))
    assert np.array_equal(spaces.vector_subtract(v2, v1), np.array([3, 3, 3]))


def test_scalar_multiply_and_dot():
    v = np.array([1, 2, 3])
    assert np.array_equal(spaces.scalar_multiply(2, v), np.array([2, 4, 6]))
    assert spaces.dot_product(v, v) == 14


def test_norms_and_unit_vector():
    v = np.array([3.0, 4.0])
    assert spaces.l2_norm(v) == 5.0
    assert spaces.l1_norm(v) == 7.0
    assert spaces.linf_norm(v) == 4.0
    unit = spaces.unit_vector(v)
    assert np.allclose(unit, v / 5.0)


def test_zscore():
    v = np.array([1.0, 1.0, 1.0])
    assert np.allclose(spaces.zscore(v), np.zeros_like(v))


def test_covariance_matrix_and_svd():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    cov = spaces.covariance_matrix(X)
    expected = np.cov(X, rowvar=False, bias=False)
    assert np.allclose(cov, expected)

    U, S, Vt = spaces.compute_svd(X)
    reconstructed = U @ np.diag(S) @ Vt
    assert np.allclose(reconstructed, X)


def test_basic_distances():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])

    assert np.isclose(spaces.euclidean_distance(v1, v2), np.sqrt(27.0))
    assert spaces.manhattan_distance(v1, v2) == 9.0
    assert np.isclose(spaces.minkowski_distance(v1, v2, p=3), (3 ** 3 * 3) ** (1 / 3))
    assert spaces.chebyshev_distance(v1, v2) == 3.0


def test_mahalanobis_and_weighted_minkowski():
    v1 = np.array([1.0, 2.0])
    v2 = np.array([2.0, 4.0])
    cov = np.eye(2)
    assert np.isclose(spaces.mahalanobis_distance(v1, v2, cov), np.sqrt(5.0))

    w = np.array([1.0, 2.0])
    expected = np.sqrt(1.0 * 1.0 + 2.0 * 4.0)
    assert np.isclose(spaces.weighted_minkowski_distance(v1, v2, p=2, w=w), expected)


def test_distribution_metrics():
    p = np.array([0.5, 0.5])
    q = np.array([0.9, 0.1])
    h = spaces.hellinger_distance(p, q)
    kl = spaces.kl_divergence(p, q)
    expected_kl = 0.5 * (np.log(0.5 / 0.9) + np.log(0.5 / 0.1))
    assert np.isclose(kl, expected_kl)
    assert h > 0


def test_other_similarities():
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.0, 2.0, 3.0])

    assert np.isclose(spaces.pearson_similarity(v1, v2), 1.0)
    assert spaces.angular_distance(v1, v2) == 0.0

    b1 = np.array([1, 0, 1, 0])
    b2 = np.array([1, 1, 0, 0])
    assert np.isclose(spaces.jaccard_similarity(b1, b2), 1 / 3)

