from setuptools import setup, find_packages

setup(
    name='QLang',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        'matplotlib',
        'fuzzywuzzy',
        'pytest',
        'tqdm',
        'python-Levenshtein',
        'scikit-learn'
    ],
    extras_require={
        'bert': ['transformers', 'torch', 'wordfreq'],
        'lsa': ['gensim']
    },
    include_package_data=True,  # Asegúrate de incluir esto
    package_data={
        'QLang': ['resources/*']  # Incluye todos los archivos en la carpeta resources
    }
    # Más configuraciones como author, license, etc.
)
