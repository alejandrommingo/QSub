# QSub Library

## Overview

The `QSub` Library is a Python package designed for creating and manipulating semantic subspaces. It leverages advanced computational models to explore and experiment with different aspects of language processing. This library is particularly useful for researchers and practitioners working in the fields of computational linguistics, cognitive psychology, and data science.

## Features

- **Semantic Space Construction**: Utilize models like LSA (Latent Semantic Analysis), Word2Vec, and BERT to construct semantic spaces.
- **Contour Generation**: Generate semantic contours for given terms within these spaces.
- **Subspace Creation**: Develop and manipulate subspaces based on semantic contours.

## Installation

To install QSub, run the following commands:

```bash
git clone https://github.com/alejandrommingo/QSub.git
cd QSub
pip install .
```

## Usage

QSub is in its first alpha version. For now, the function for extracting contours from the [Gallito](https://psicoee.uned.es/quantumlikespace/especifications/ASSE_searchBySimpleProjectionInGTFS.aspx) program's API has been developed. Future updates will be announced.

```python
from QSub.contours import gallito_contour

# Parameters
word = "example"
api_key = "your_api_key"  # Replace with your Gallito API key
space_name = "quantumlikespace_spanish"
neighbors = 100

# Generate contour
contour = gallito_contour(word, api_key, space_name, neighbors)
```

Remember to replace `"your_api_key"` with your actual Gallito API key.

## Contributing

We welcome contributions to the QSub Library! If you have suggestions for improvements or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [GNU General Public License](LICENSE).
