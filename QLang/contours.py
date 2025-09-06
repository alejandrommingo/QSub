"""
QSub Contours Module - Clean Version

This module contains essential functions for contextual contour analysis.
"""

import requests
import re
import numpy as np
import matplotlib.pyplot as plt
from QSub.semantic_spaces import get_static_word_vector, get_contextual_word_vector


def get_complete_contextual_contour_wikipedia(
    word,
    model_name="bert-base-uncased",
    max_contexts=None,
    include_static=True,
    verbose=False
):
    """
    Extract complete contextual contour from Wikipedia processing ALL occurrences.
    
    This function extracts ALL available contexts from Wikipedia and processes
    ALL occurrences of the target word, providing comprehensive analysis.
    
    Parameters
    ----------
    word : str
        Target term to search on Wikipedia
    model_name : str, default="bert-base-uncased"
        Model for contextual embeddings
    max_contexts : int, optional
        Maximum contexts to process. None = ALL contexts
    include_static : bool, default=True
        Whether to include static word vector
    verbose : bool, default=False
        Whether to show processing details
        
    Returns
    -------
    dict or None
        Dictionary with occurrence vectors and metadata, or None if failed
    """
    # Headers for Wikipedia API (using the exact format that worked in terminal)
    headers = {
        'User-Agent': 'QSub/1.0 (https://github.com/alejandrommingo/QSub; contact@example.com) Python/requests'
    }
    
    try:
        # Use Wikipedia public API
        if verbose:
            print(f"Querying Wikipedia API for: {word}")
        
        api_url = "https://en.wikipedia.org/w/api.php"
        params = {
            'action': 'query',
            'prop': 'extracts',
            'explaintext': True,
            'titles': word,
            'format': 'json',
        }
        
        if verbose:
            print(f"API URL: {api_url}")
            print(f"Params: {params}")
            print(f"Headers: {headers}")
        
        response = requests.get(api_url, params=params, headers=headers, timeout=15)
        
        if verbose:
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code != 200:
            if verbose:
                print(f"Wikipedia API error: {response.status_code}")
                print(f"Response text: {response.text[:500]}")
            return None
        
        data = response.json()
        pages = data.get('query', {}).get('pages', {})
        full_text = ""
        
        for page_data in pages.values():
            if 'extract' in page_data:
                full_text = page_data['extract']
                break
        
        if not full_text:
            if verbose:
                print("No content found in Wikipedia")
            return None
        
        if verbose:
            print(f"Extracted text: {len(full_text)} characters")
        
        # Split into sentences
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = [s.strip() for s in re.split(sentence_pattern, full_text) if s.strip()]
        
        if verbose:
            print(f"Total sentences: {len(sentences)}")
        
        # Find sentences containing the target word
        word_lower = word.lower()
        valid_contexts = []
        
        for sentence in sentences:
            if re.search(r'\b' + re.escape(word_lower) + r'\b', sentence.lower()):
                if len(sentence.split()) >= 5:  # Minimum sentence length
                    valid_contexts.append(sentence)
        
        if verbose:
            print(f"Valid contexts found: {len(valid_contexts)}")
        
        # Apply context limit if specified
        if max_contexts is not None:
            valid_contexts = valid_contexts[:max_contexts]
            if verbose:
                print(f"Processing {len(valid_contexts)} contexts (limited)")
        else:
            if verbose:
                print(f"Processing ALL {len(valid_contexts)} contexts")
        
        # Process ALL occurrences in each context
        results = {}
        total_occurrences = 0
        
        for ctx_idx, context in enumerate(valid_contexts):
            words = context.split()
            term_positions = []
            
            # Find ALL positions of the target word
            for word_idx, context_word in enumerate(words):
                clean_word = re.sub(r'[^\w]', '', context_word.lower())
                if clean_word == word_lower:
                    term_positions.append(word_idx)
            
            # Process each occurrence
            for occ_idx, pos in enumerate(term_positions):
                try:
                    # Create marked context
                    context_words = words.copy()
                    context_words[pos] = f"[TARGET]{context_words[pos]}[/TARGET]"
                    marked_context = ' '.join(context_words)
                    
                    # Get contextual embedding
                    vector = get_contextual_word_vector(
                        term=words[pos],
                        text=marked_context,
                        model_name=model_name
                    )
                    
                    if vector is not None:
                        occurrence_key = f"ctx_{ctx_idx:03d}_occ_{occ_idx:02d}"
                        results[occurrence_key] = {
                            'vector': vector,
                            'context': context,
                            'position': pos,
                            'word': words[pos],
                            'context_index': ctx_idx,
                            'occurrence_index': occ_idx
                        }
                        total_occurrences += 1
                
                except (ValueError, TypeError, AttributeError) as e:
                    if verbose:
                        print(f"Error processing occurrence at position {pos}: {e}")
        
        # Add static vector if requested
        if include_static:
            try:
                static_vector = get_static_word_vector(word=word, model_name=model_name)
                if static_vector is not None:
                    results['static_vector'] = {
                        'vector': static_vector,
                        'context': 'static_representation',
                        'position': -1,
                        'word': word,
                        'context_index': -1,
                        'occurrence_index': -1
                    }
            except (ValueError, TypeError, AttributeError) as e:
                if verbose:
                    print(f"Could not get static vector: {e}")
        
        if verbose:
            print(f"Processing complete. Total occurrences: {total_occurrences}")
            print(f"Vectors generated: {len(results)}")
        
        return results if results else None
        
    except (requests.RequestException, ValueError, KeyError) as e:
        if verbose:
            print(f"Error in Wikipedia extraction: {e}")
        return None


def analyze_contextual_contour(contour_dict, target_word, include_static=True):
    """
    Analyze similarity patterns in a contextual contour.
    
    Parameters
    ----------
    contour_dict : dict
        Dictionary with contour data from get_complete_contextual_contour_wikipedia
    target_word : str
        Target word being analyzed
    include_static : bool, default=True
        Whether to include static vector in analysis
        
    Returns
    -------
    dict
        Dictionary with analysis results
    """
    if not contour_dict:
        return None
    
    # Separate contextual and static vectors
    contextual_vectors = []
    static_vector = None
    
    for key, data in contour_dict.items():
        if isinstance(data, dict) and 'vector' in data:
            if key == 'static_vector':
                static_vector = data['vector']
            else:
                contextual_vectors.append(data['vector'])
    
    if not contextual_vectors:
        return None
    
    n_contextual = len(contextual_vectors)
    
    # Calculate pairwise similarities between contextual vectors
    similarities = []
    for i in range(n_contextual):
        for j in range(i + 1, n_contextual):
            sim = np.dot(contextual_vectors[i], contextual_vectors[j]) / (
                np.linalg.norm(contextual_vectors[i]) * np.linalg.norm(contextual_vectors[j])
            )
            similarities.append(sim)
    
    # Calculate statistics
    similarities = np.array(similarities)
    mean_similarity = np.mean(similarities)
    similarity_std = np.std(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    
    results = {
        'target_word': target_word,
        'n_contextual_vectors': n_contextual,
        'mean_contextual_similarity': mean_similarity,
        'similarity_std': similarity_std,
        'min_similarity': min_similarity,
        'max_similarity': max_similarity,
        'similarity_range': max_similarity - min_similarity
    }
    
    # Static vector analysis
    if include_static and static_vector is not None:
        static_similarities = []
        for vector in contextual_vectors:
            sim = np.dot(static_vector, vector) / (
                np.linalg.norm(static_vector) * np.linalg.norm(vector)
            )
            static_similarities.append(sim)
        
        static_similarities = np.array(static_similarities)
        results.update({
            'has_static': True,
            'mean_static_similarity': np.mean(static_similarities),
            'static_similarities': static_similarities,
            'static_similarity_std': np.std(static_similarities)
        })
    else:
        results['has_static'] = False
    
    return results


def visualize_contextual_contour(analysis_results, save_path=None):
    """
    Create visualizations for contextual contour analysis.
    
    Parameters
    ----------
    analysis_results : dict
        Results from analyze_contextual_contour
    save_path : str, optional
        Path to save the figure
        
    Returns
    -------
    matplotlib.figure.Figure
        Generated figure
    """
    if not analysis_results:
        return None
    
    has_static = analysis_results.get('has_static', False)
    
    # Create figure with appropriate subplots
    if has_static:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Plot 1: Similarity distribution
    target_word = analysis_results['target_word']
    mean_sim = analysis_results['mean_contextual_similarity']
    sim_std = analysis_results['similarity_std']
    
    ax1.hist([mean_sim], bins=1, alpha=0.7, color='steelblue', 
             label=f'Mean: {mean_sim:.3f}')
    ax1.axvline(mean_sim, color='red', linestyle='--', 
                label=f'Mean ± Std: {mean_sim:.3f} ± {sim_std:.3f}')
    ax1.axvline(mean_sim - sim_std, color='red', linestyle=':', alpha=0.5)
    ax1.axvline(mean_sim + sim_std, color='red', linestyle=':', alpha=0.5)
    
    ax1.set_title(f'Contextual Similarity Distribution\n"{target_word}"')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Static vs contextual similarities (if available)
    if has_static:
        static_sims = analysis_results['static_similarities']
        context_indices = range(1, len(static_sims) + 1)
        
        ax2.bar(context_indices, static_sims, color='orange', alpha=0.7)
        ax2.set_title(f'Static vs Contextual Similarities\n"{target_word}"')
        ax2.set_xlabel('Context Index')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_ylim(0, 1)
        
        # Add mean line
        mean_static = analysis_results['mean_static_similarity']
        ax2.axhline(y=mean_static, color='red', linestyle='--', 
                   label=f'Mean: {mean_static:.3f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig


def cosine_similarity(v1, v2):
    """
    Calculate cosine similarity between two vectors.
    
    Parameters
    ----------
    v1, v2 : array-like
        Input vectors
        
    Returns
    -------
    float
        Cosine similarity
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
