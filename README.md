# QSub Library

## Overview

The `QSub` Library is a Python package designed for creating and manipulating semantic subspaces. It leverages advanced computational models to explore and experiment with different aspects of language processing. This library is particularly useful for researchers and practitioners working in the fields of computational linguistics, cognitive psychology, and data science.

## Features

- **Semantic Space Construction**: Utilize models like LSA (Latent Semantic Analysis), Word2Vec, GloVe, GPT2, ELMo and BERT variants (including DistilBERT) to construct semantic spaces. Helper functions allow extracting vectors from selected layers when supported.
- **Contour Generation**: Generate semantic contours for given terms within these spaces.
- **Subspace Creation**: Develop and manipulate subspaces based on semantic contours.

## Installation

To install QSub, run the following commands:

```bash
git clone https://github.com/alejandrommingo/QSub.git
cd QSub
pip install .
```

The BERT utilities are optional. Install the extra dependencies with:

```bash
pip install .[bert]
```

The BERT helper functions accept an ``output_layer`` argument to select a
specific hidden layer (by index), ``"last"`` for the final layer (default) or
``"all"`` to obtain a matrix with all layers.

## Usage

QSub is in its first alpha version. You can find the first use example in our [Conceptual Contour Notebook](https://github.com/alejandrommingo/QSub/blob/main/notebooks/QSub_conceptual_contour_example.ipynb)

![Deserved Neighbors for Conceptual Contour of Palestine](https://github.com/alejandrommingo/QSub/blob/main/img/QSub_conceptual_contour_example.png)

Remember to replace `"API_KEY"` with your actual Gallito API key.

## Contributing

We welcome contributions to the QSub Library! If you have suggestions for improvements or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [GNU General Public License](LICENSE).

## Citing this repository

If you use this repository in your research, please cite it as follows:

Martinez-Mingo, A. (2024). *QSub: Quantum Subspaces*. GitHub repository. Available at: https://github.com/alejandrommingo/QSub
