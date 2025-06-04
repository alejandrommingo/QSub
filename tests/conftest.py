import sys
import types

import numpy as np
import pytest

import QSub.contours as contours
import QSub.semantic_spaces as spaces

@pytest.fixture(autouse=True)
def stub_gallito(monkeypatch):
    def dummy_neighbors(word, code, space, neighbors=100):
        return {f"{word}_{i}": np.random.rand(300) for i in range(neighbors)}

    def dummy_word_vector(word, code, space):
        return np.random.rand(300)

    def dummy_lsa_corpus(path, code, space):
        with open(path, 'r', encoding='utf-8') as f:
            terms = [line.strip() for line in f]
        return {term: np.random.rand(300) for term in terms}

    def dummy_superterm(path, code, space):
        return np.random.rand(300), np.random.rand(10)

    monkeypatch.setattr(contours, "get_neighbors_matrix_gallito", dummy_neighbors)
    monkeypatch.setattr(contours, "get_superterm_gallito", dummy_superterm)
    monkeypatch.setattr(spaces, "get_word_vector_gallito", dummy_word_vector)
    monkeypatch.setattr(spaces, "get_lsa_corpus_gallito", dummy_lsa_corpus)

    # Stub wordcloud module if not installed
    if "wordcloud" not in sys.modules:
        dummy_wc = types.ModuleType("wordcloud")
        dummy_wc.WordCloud = object
        sys.modules["wordcloud"] = dummy_wc
    yield

