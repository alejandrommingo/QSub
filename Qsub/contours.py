import requests
import pandas as pd
import xml.etree.ElementTree as ET


def gallito_contour(word, gallito_code, space_name, neighbors=100, min_cosine_contour=0.3):
    k = 300  # Define K dimensions
    n = neighbors  # Define N neighbors
    df = pd.DataFrame()

    # Preparar la consulta a la API de Gallito
    query_xml = f"""
    <getNearestNeighboursList xmlns='http://tempuri.org/'>
        <code>{gallito_code}</code>
        <a>{word}</a>
        <txtNegate></txtNegate>
        <n>{n}</n>
        <lenght_biased>false</lenght_biased>
    </getNearestNeighboursList>
    """

    space_url = f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getNearestNeighboursList"
    response = requests.post(space_url, data=query_xml, headers={'Content-Type': 'text/xml'})

    # Procesar la respuesta
    root = ET.fromstring(response.content)
    terms = [term.text for term in root.findall('.//term')]

    for term in terms:
        vector_url = f"http://psicoee.uned.es/{space_name}/Service.svc/webHttp/getVectorOfTerm?code={gallito_code}&a={term}"
        vector_response = requests.get(vector_url)
        vector_root = ET.fromstring(vector_response.content)
        vector = [float(dim.text) for dim in vector_root.findall('.//dim')]

        # Añadir el vector al DataFrame
        df[term] = vector

    # Filtrar por el mínimo coseno
    df = df.loc[:, df.corr().iloc[0] > min_cosine_contour]

    return df
