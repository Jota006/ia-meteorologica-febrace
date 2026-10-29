import requests
import json
import os
from datetime import datetime

# Coordenadas exatas de Jales - SP (Escola Cel de Jales)
LATITUDE = -20.2668
LONGITUDE = -50.5458

def atualizar_clima():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={LATITUDE}&longitude={LONGITUDE}&current=temperature_2m,relative_humidity_2m"
    try:
        resposta = requests.get(url)
        resposta.raise_for_status()  # Garante que deu tudo certo com a requisição
        dados = resposta.json()
        
        # Aqui você continua a manipulação dos dados meteorológicos
        print("Dados obtidos com sucesso:", dados)
        return dados

    except Exception as e:
        print(f"Erro ao buscar os dados do clima: {e}")
        return None

if __name__ == "__main__":
    atualizar_clima()
