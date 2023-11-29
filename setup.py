from setuptools import setup, find_packages

setup(
    name='QSub',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'requests',
        're',
        'matplotlib',
        'fuzzywuzzy',
        'html',
        'xml.etree.ElementTree'
        'transformers',
        'gensim',
        'pytest'
    ],
    include_package_data=True,  # Asegúrate de incluir esto
    package_data={
        'QSub': ['resources/*']  # Incluye todos los archivos en la carpeta resources
    }
    # Más configuraciones como author, license, etc.
)
