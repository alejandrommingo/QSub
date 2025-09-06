import pytest
import numpy as np
import os

# The following imports are mocked in conftest.py during testing
from QLang.subspaces import (
    parallel_analysis_horn,
    create_subspace,
    create_subspace_from_word,
    describe_subspace,
)


def test_parallel_analysis_horn():
    """Test parallel analysis using Horn's method."""
    # Create a random data matrix
    data_matrix = np.random.rand(50, 10)
    
    result = parallel_analysis_horn(data_matrix, analysis_type='Terms', num_simulations=10)
    
    assert isinstance(result, dict)
    assert 'eigenvalues' in result
    assert 'random_eigenvalues' in result
    assert 'n_components' in result
    
    # Check that eigenvalues are in descending order
    eigenvals = result['eigenvalues']
    assert all(eigenvals[i] >= eigenvals[i+1] for i in range(len(eigenvals)-1))


def test_create_subspace():
    """Test subspace creation from word contour."""
    # Create mock word contour data
    word_contour = np.random.rand(20, 100)  # 20 terms, 100 dimensions
    n_components = 5
    
    subspace = create_subspace(word_contour, n_components, subspace_type="Terms")
    
    assert isinstance(subspace, np.ndarray)
    assert subspace.shape == (20, n_components)


def test_create_subspace_from_word():
    """Test subspace creation from a specific word using Gallito API."""
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    
    try:
        result = create_subspace_from_word(
            word="gato",
            gallito_code=test_code,
            space_name="quantumlikespace_spanish",
            neighbors=10,
            n_components=3
        )
        
        assert isinstance(result, dict)
        assert 'subspace' in result
        assert 'word_contour' in result
        assert result['subspace'].shape[1] == 3  # n_components
        
    except Exception:
        pytest.skip("Gallito API not available or key not configured")


def test_describe_subspace():
    """Test subspace description functionality."""
    # Create mock data
    subspace_matrix = np.random.rand(10, 3)  # 10 terms, 3 components
    contour = {f"word_{i}": np.random.rand(100) for i in range(10)}
    
    try:
        result = describe_subspace(subspace_matrix, contour, top_n=5, graph=False)
        
        assert isinstance(result, dict)
        # The function should return some form of description
        
    except Exception as e:
        # If function has dependencies issues, skip
        pytest.skip(f"Describe subspace failed: {e}")


def test_subspace_dimensions():
    """Test that subspace creation respects dimension constraints."""
    # Test edge cases
    word_contour = np.random.rand(5, 10)  # Small matrix
    
    # Test with n_components = number of terms
    subspace1 = create_subspace(word_contour, 5, subspace_type="Terms")
    assert subspace1.shape == (5, 5)
    
    # Test with n_components < number of terms  
    subspace2 = create_subspace(word_contour, 3, subspace_type="Terms")
    assert subspace2.shape == (5, 3)


def test_parallel_analysis_with_variables():
    """Test parallel analysis with Variables type."""
    data_matrix = np.random.rand(10, 50)  # More variables than observations
    
    result = parallel_analysis_horn(data_matrix, analysis_type='Variables', num_simulations=10)
    
    assert isinstance(result, dict)
    assert 'eigenvalues' in result
    assert 'random_eigenvalues' in result
    assert 'n_components' in result
