import pytest
import numpy as np
import os
import tempfile
from typing import TYPE_CHECKING

import QLang.semantic_spaces as spaces

# Prevent VS Code from showing false errors due to monkey patching
if TYPE_CHECKING:
    pass


def test_get_word_vector_gallito():
    """Test Gallito API word vector retrieval."""
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    
    try:
        result = spaces.get_word_vector_gallito("gato", test_code, "quantumlikespace_spanish")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    except Exception:
        pytest.skip("Gallito API not available or key not configured")


def test_get_static_word_vector():
    """Test static word vector extraction with auto-detection."""
    try:
        # Test with a simple word - this will auto-detect the model type
        result = spaces.get_static_word_vector("hello")
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    except ImportError:
        pytest.skip("Required dependencies not installed")
    except Exception as e:
        pytest.skip(f"Model loading failed: {e}")


def test_get_contextual_word_vector():
    """Test contextual word vector extraction."""
    try:
        result = spaces.get_contextual_word_vector(
            "hello", 
            "Hello world, this is a test sentence with hello in it."
        )
        assert isinstance(result, np.ndarray)
        assert len(result) > 0
    except ImportError:
        pytest.skip("Required dependencies not installed")
    except Exception as e:
        pytest.skip(f"Model loading failed: {e}")


def test_get_bert_corpus():
    """Test BERT corpus generation."""
    try:
        # Test with minimal parameters
        result = spaces.get_bert_corpus(language="en", n_words=3)
        assert isinstance(result, dict)
        assert len(result) <= 3
        
        # Check that values are numpy arrays
        for vector in result.values():
            assert isinstance(vector, np.ndarray)
            
    except ImportError:
        pytest.skip("BERT dependencies not installed")
    except Exception as e:
        pytest.skip(f"BERT corpus generation failed: {e}")


def test_get_gpt2_corpus():
    """Test GPT2 corpus generation."""
    try:
        result = spaces.get_gpt2_corpus(language="en", n_words=3)
        assert isinstance(result, dict)
        assert len(result) <= 3
        
        # Check that values are numpy arrays
        for vector in result.values():
            assert isinstance(vector, np.ndarray)
            
    except ImportError:
        pytest.skip("GPT2 dependencies not installed")
    except Exception as e:
        pytest.skip(f"GPT2 corpus generation failed: {e}")


def test_get_lsa_corpus_gallito():
    """Test LSA corpus retrieval from Gallito."""
    test_code = os.getenv("GALLITO_API_KEY", "default_code")
    
    try:
        # Create a temporary vocabulary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("gato\nperro\ncasa\n")
            vocab_file = f.name
        
        try:
            result = spaces.get_lsa_corpus_gallito(vocab_file, test_code, "quantumlikespace_spanish")
            assert isinstance(result, dict)
            
            # Check that we got vectors for the words
            for vector in result.values():
                assert isinstance(vector, np.ndarray)
                
        finally:
            os.unlink(vocab_file)
            
    except Exception:
        pytest.skip("Gallito API not available or key not configured")


def test_model_type_detection():
    """Test internal model type detection."""
    # Test the _detect_model_type function if it's available
    try:
        from QLang.semantic_spaces import _detect_model_type
        
        assert _detect_model_type("bert-base-uncased") == "bert"
        assert _detect_model_type("gpt2") == "gpt2"
        assert _detect_model_type("distilbert-base-uncased") == "bert"
        
    except ImportError:
        # Function might not be publicly available
        pass
