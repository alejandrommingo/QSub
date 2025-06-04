import os
import sys
import types

import numpy as np
import QSub.contours as contours

# Ensure wordcloud dependency is available
sys.modules.setdefault("wordcloud", types.SimpleNamespace(WordCloud=object))

from QSub.subspaces import parallel_analysis_horn, create_subspace

def test_parallel_analysis_horn():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    # Ejecutamos la función
    contour = contours.get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)
    resultado = parallel_analysis_horn(contour)

    # Comprobamos los tests
    assert isinstance(resultado, np.int64)
    

def test_create_subspace():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    # Ejecutamos la función
    contour = contours.get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)
    resultado = create_subspace(contour, 2)

    # Comprobamos los tests
    assert isinstance(resultado, np.ndarray)


