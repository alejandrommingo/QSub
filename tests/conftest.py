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

    def dummy_word_vector_bert(word, model_name="bert-base-uncased", output_layer="last"):
        if output_layer == "all":
            return np.random.rand(12, 768)
        return np.random.rand(768)

    def dummy_word_vector_gpt2(word, model_name="gpt2", output_layer="last"):
        if output_layer == "all":
            return np.random.rand(12, 768)
        return np.random.rand(768)

    def dummy_gpt2_corpus(language="en", model_name="gpt2", n_words=1000, output_layer="last"):
        if output_layer == "all":
            return {f"{language}_{i}": np.random.rand(12, 768) for i in range(n_words)}
        return {f"{language}_{i}": np.random.rand(768) for i in range(n_words)}

    def dummy_word_vector_word2vec(word, model_name="word2vec-google-news-300"):
        return np.random.rand(300)

    def dummy_word2vec_corpus(language="en", model_name="word2vec-google-news-300", n_words=1000):
        return {f"{language}_{i}": np.random.rand(300) for i in range(n_words)}

    def dummy_word_vector_glove(word, model_name="glove-wiki-gigaword-300"):
        return np.random.rand(300)

    def dummy_glove_corpus(language="en", model_name="glove-wiki-gigaword-300", n_words=1000):
        return {f"{language}_{i}": np.random.rand(300) for i in range(n_words)}

    def dummy_word_vector_elmo(word, model_name="small", output_layer="average"):
        if output_layer == "all":
            return np.random.rand(3, 1024)
        return np.random.rand(1024)

    def dummy_elmo_corpus(language="en", model_name="small", n_words=1000, output_layer="average"):
        if output_layer == "all":
            return {f"{language}_{i}": np.random.rand(3, 1024) for i in range(n_words)}
        return {f"{language}_{i}": np.random.rand(1024) for i in range(n_words)}

    def dummy_word_vector_distilbert(word, model_name="distilbert-base-uncased", output_layer="last"):
        return np.random.rand(768)

    def dummy_distilbert_corpus(language="en", model_name="distilbert-base-uncased", n_words=1000, output_layer="last"):
        return {f"{language}_{i}": np.random.rand(768) for i in range(n_words)}

    def dummy_bert_corpus(
        language="en",
        model_name="bert-base-uncased",
        n_words=1000,
        output_layer="last",
    ):
        if output_layer == "all":
            return {f"{language}_{i}": np.random.rand(12, 768) for i in range(n_words)}
        return {f"{language}_{i}": np.random.rand(768) for i in range(n_words)}

    monkeypatch.setattr(contours, "get_neighbors_matrix_gallito", dummy_neighbors)
    monkeypatch.setattr(contours, "get_superterm_gallito", dummy_superterm)
    monkeypatch.setattr(spaces, "get_word_vector_gallito", dummy_word_vector)
    monkeypatch.setattr(spaces, "get_lsa_corpus_gallito", dummy_lsa_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_bert", dummy_word_vector_bert)
    monkeypatch.setattr(spaces, "get_bert_corpus", dummy_bert_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_gpt2", dummy_word_vector_gpt2)
    monkeypatch.setattr(spaces, "get_gpt2_corpus", dummy_gpt2_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_word2vec", dummy_word_vector_word2vec)
    monkeypatch.setattr(spaces, "get_word2vec_corpus", dummy_word2vec_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_glove", dummy_word_vector_glove)
    monkeypatch.setattr(spaces, "get_glove_corpus", dummy_glove_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_elmo", dummy_word_vector_elmo)
    monkeypatch.setattr(spaces, "get_elmo_corpus", dummy_elmo_corpus)
    monkeypatch.setattr(spaces, "get_word_vector_distilbert", dummy_word_vector_distilbert)
    monkeypatch.setattr(spaces, "get_distilbert_corpus", dummy_distilbert_corpus)

    # Stub wordcloud module if not installed
    if "wordcloud" not in sys.modules:
        dummy_wc = types.ModuleType("wordcloud")
        dummy_wc.WordCloud = object
        sys.modules["wordcloud"] = dummy_wc
    yield

