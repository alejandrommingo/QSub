from setuptools import setup, find_packages

setup(
    name='QSub',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        'transformers',
        'gensim',
        'pytest'
    ],
    # Más configuraciones como author, license, etc.
)
