import requests
import json
import base64
from PIL import Image
import io

# Fazer uma requisição para análise de tema
data = {
    "theme": "vacinação covid",
    "max_posts": 10,
    "language": "pt"
}

response = requests.post(
    "http://localhost:8000/analyze-theme",
    json=data
)

if response.status_code == 200:
    result = response.json()
    print(f"Tema: {result['theme']}")
    print(f"Total de posts: {result['total_posts']}")
    print(f"Fake news detectadas: {result['fake_news_count']}")
    print(f"Resumo: {json.dumps(result['analysis_summary'], indent=2)}")
    
    # Se quiser visualizar a imagem do dashboard
    if result['dashboard_image']:
        image_data = base64.b64decode(result['dashboard_image'])
        image = Image.open(io.BytesIO(image_data))
        image.show()  # Isso abrirá a imagem no visualizador padrão
else:
    print(f"Erro: {response.status_code} - {response.text}")