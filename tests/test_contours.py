import pytest
import numpy as np
import os
import tempfile

import QLang.contours as contours


def test_cosine_similarity():
    """Test basic cosine similarity calculation."""
    v1 = np.array([1, 0, 0])
    v2 = np.array([0, 1, 0])
    v3 = np.array([1, 0, 0])
    
    # Test orthogonal vectors
    assert abs(contours.cosine_similarity(v1, v2)) < 1e-10
    
    # Test identical vectors  
    assert abs(contours.cosine_similarity(v1, v3) - 1.0) < 1e-10


def test_analyze_contextual_contour():
    """Test contextual contour analysis."""
    # Create mock data
    mock_contour = {
        'contextual_vectors': [
            np.array([1, 0, 0]),
            np.array([0, 1, 0]),
            np.array([1, 1, 0]) / np.sqrt(2)
        ],
        'contexts': ['context1', 'context2', 'context3']
    }
    
    result = contours.analyze_contextual_contour(mock_contour, 'test_word')
    
    assert 'similarity_matrix' in result
    assert 'average_similarity' in result
    assert 'std_similarity' in result
    assert result['similarity_matrix'].shape == (3, 3)


@pytest.mark.slow
def test_get_complete_contextual_contour_wikipedia():
    """Test Wikipedia contextual contour extraction."""
    # This is an integration test that requires internet
    # We'll test with a simple word and minimal parameters
    try:
        result = contours.get_complete_contextual_contour_wikipedia(
            'cat', 
            max_contexts=2,
            verbose=False
        )
        
        # Check that we got some results (function returns dict with occurrence keys)
        assert result is not None
        assert isinstance(result, dict)
        assert len(result) <= 2  # Should have at most 2 contexts
        
    except (ConnectionError, TimeoutError, ValueError) as e:
        # If network/API fails, skip the test
        pytest.skip(f"Wikipedia API test failed: {e}")


def test_visualize_contextual_contour():
    """Test visualization function."""
    # Create mock analysis results
    mock_results = {
        'target_word': 'test_word',
        'similarity_matrix': np.array([[1.0, 0.5], [0.5, 1.0]]),
        'average_similarity': 0.75,
        'std_similarity': 0.25,
        'contexts': ['context1', 'context2']
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_plot.png')
        
        # Should not raise an exception
        try:
            contours.visualize_contextual_contour(mock_results, save_path=save_path)
            # Check that file was created
            assert os.path.exists(save_path)
        except ImportError:
            # Skip if matplotlib not available
            pytest.skip("Matplotlib not available for visualization test")
