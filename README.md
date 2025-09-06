# QLang Library

## Overview

The **QLang** (Quantum Language) Library is a Python package designed for creating and analyzing contextual semantic subspaces. It leverages advanced computational models and real-world data sources to explore how word meanings vary across different contexts. This library is particularly useful for researchers working in computational linguistics, semantic analysis, and natural language processing.

QLang provides tools for extracting contextual instances of words from Wikipedia, analyzing semantic variations, and visualizing how meanings shift across different linguistic environments.

## Key Features

- **Contextual Contour Analysis**: Extract and analyze how words behave semantically across multiple real-world contexts from Wikipedia
- **Semantic Space Construction**: Utilize models like LSA (Latent Semantic Analysis), Word2Vec, GloVe, GPT2, ELMo and BERT variants (including DistilBERT) to construct semantic spaces
- **Wikipedia Integration**: Automatic extraction of contextual instances with robust error handling and User-Agent compliance
- **Advanced Visualizations**: Generate comprehensive plots including similarity matrices, t-SNE projections, and statistical distributions
- **Subspace Creation**: Develop and manipulate subspaces based on semantic contours
- **Statistical Analysis**: Detailed metrics for semantic coherence, variability, and polysemy detection

## Installation

To install QLang, run the following commands:

```bash
git clone https://github.com/alejandrommingo/QSub.git
cd QSub
pip install -r requirements.txt
pip install .
```

### Optional Dependencies

The BERT utilities are optional. Install the extra dependencies with:

```bash
pip install .[bert]
```

The BERT helper functions accept an ``output_layer`` argument to select a
specific hidden layer (by index), ``"last"`` for the final layer (default) or
``"all"`` to obtain a matrix with all layers.

### Required Dependencies

- numpy, scipy, sklearn (core computation)
- matplotlib, seaborn (visualization)
- requests (Wikipedia API access)
- pandas (data manipulation)
- transformers, torch (optional, for BERT models)

## Usage & Examples

QLang provides three main demonstration notebooks showcasing different aspects of the library:

### 1. [Contextual Contours](notebooks/Contextual_Contours.ipynb) 
Complete pipeline demonstrating:
- Automatic extraction of contextual instances from Wikipedia
- Comprehensive similarity analysis and statistical metrics
- Advanced visualizations (similarity matrices, t-SNE, PCA)
- Detailed interpretation of semantic variability patterns

### 2. [Semantic Spaces](notebooks/Semantic_Spaces.ipynb)
Focus on semantic space construction:
- Multiple embedding models (Word2Vec, BERT, GloVe)
- Layer-specific analysis for transformer models
- Semantic space manipulation and exploration

### 3. [Conceptual Contour LSA](notebooks/Conceptual_Contour_LSA.ipynb)
Original conceptual demonstration:
- Basic contour generation concepts using LSA
- Foundational semantic analysis approaches

![Contextual Contour Analysis Example](https://github.com/alejandrommingo/QSub/blob/main/img/QSub_conceptual_contour_example.png)

### Quick Start

```python
from QLang import contours

# Extract contextual contour from Wikipedia
target_word = "bank"
contextual_vectors, contexts = contours.get_complete_contextual_contour_wikipedia(
    target_word, 
    max_articles=5,
    window_size=5
)

# Analyze semantic variations
analysis_results = contours.analyze_contextual_contour(
    contextual_vectors, 
    contexts
)

# Generate comprehensive visualizations
contours.visualize_contextual_contour(
    analysis_results['similarity_matrix'],
    contexts,
    target_word
)
```

## Module Structure

- **`contours.py`**: Core module for contextual contour analysis
  - `get_complete_contextual_contour_wikipedia()`: Extract contextual instances from Wikipedia
  - `analyze_contextual_contour()`: Compute similarity matrices and statistics
  - `visualize_contextual_contour()`: Generate comprehensive visualizations

- **`semantic_spaces.py`**: Semantic space construction and manipulation
  - Support for multiple embedding models (Word2Vec, BERT, GloVe, etc.)
  - Layer-specific extraction for transformer models

- **`subspaces.py`**: Subspace creation and analysis tools

## Applications

QLang is designed for research in:
- **Polysemy Detection**: Identify words with multiple meanings across contexts
- **Semantic Drift Analysis**: Study how word meanings change over time or domains
- **Contextual Semantics**: Understand how context influences word interpretation
- **Computational Linguistics**: Develop models that account for semantic variability
- **Cognitive Psychology**: Study human semantic processing and representation

## Contributing

We welcome contributions to the QLang Library! If you have suggestions for improvements or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [GNU General Public License](LICENSE).

## Citing this repository

If you use this repository in your research, please cite it as follows:

Martinez-Mingo, A. (2024). *QLang: Quantum Language Analysis*. GitHub repository. Available at: https://github.com/alejandrommingo/QSub
