import os
import numpy as np
from QSub.contours import get_neighbors_matrix_gallito
from QSub.subspaces import parallel_analysis_horn

def test_parallel_analysis_horn():
    # Parámetros de prueba
    test_word = "china"
    test_code = os.getenv('GALLITO_API_KEY', 'default_code')
    test_space_name = "quantumlikespace_spanish"
    test_neighbors = 10

    # Ejecutamos la función
    contour = get_neighbors_matrix_gallito(test_word, test_code, test_space_name, neighbors=test_neighbors)
    resultado = parallel_analysis_horn(contour)

    # Comprobamos los tests
    assert type(resultado) is np.int64 # Comprueba que el resultado es un integer