from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from io import BytesIO
import re
import json
from typing import Dict, List, Optional
from PIL import Image
import asyncio
from models.isso import test_enhanced_model  # Sua função existente
from pydantic import BaseModel

app = FastAPI(title="Fake News Analysis API")

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # URLs do frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = "your-api-key-here"

# Cache para armazenar análises (em produção use Redis ou database)
analysis_cache = {}

class ThemeAnalysisRequest(BaseModel):
    theme: str
    max_posts: Optional[int] = 10
    language: Optional[str] = 'pt'

def parse_report(report_content: str) -> Dict:
    """Parseia o relatório e extrai os dados estruturados"""
    patterns = {
        'total': r"Total de textos analisados: (\d+)",
        'fake': r"Classificados como FAKE: (\d+) \((\d+\.\d+)%\)",
        'true': r"Classificados como TRUE: (\d+) \((\d+\.\d+)%\)",
        'confidence': r"Confiança média: (\d+\.\d+)%",
        'reliability': r"Score de confiabilidade médio: (\d+\.\d+)%",
        'text': r"--- TEXTO (\d+) ---(.*?)(?=--- TEXTO|\Z)",
        'ml': r"Predição ML: (FAKE|TRUE) \(Confiança: (\d+\.\d+)%\)",
        'gpt': r"Recomendação GPT: (FAKE|TRUE|INCERTO)",
        'reliability_score': r"Score de Confiabilidade: (\d+\.\d+)%"
    }
    
    data = {'summary': {}, 'texts': []}
    
    try:
        # Dados gerais
        data['summary']['total_texts'] = int(re.search(patterns['total'], report_content).group(1))
        data['summary']['fake_count'] = int(re.search(patterns['fake'], report_content).group(1))
        data['summary']['fake_percentage'] = float(re.search(patterns['fake'], report_content).group(2))
        data['summary']['true_count'] = int(re.search(patterns['true'], report_content).group(1))
        data['summary']['true_percentage'] = float(re.search(patterns['true'], report_content).group(2))
        data['summary']['avg_confidence'] = float(re.search(patterns['confidence'], report_content).group(1))
        data['summary']['avg_reliability'] = float(re.search(patterns['reliability'], report_content).group(1))
        
        # Dados por texto
        text_matches = re.findall(patterns['text'], report_content, re.DOTALL)
        for text_num, text_content in text_matches:
            try:
                ml_match = re.search(patterns['ml'], text_content)
                gpt_match = re.search(patterns['gpt'], text_content)
                reliability_match = re.search(patterns['reliability_score'], text_content)
                
                text_data = {
                    'number': int(text_num),
                    'ml_prediction': ml_match.group(1) if ml_match else 'UNKNOWN',
                    'ml_confidence': float(ml_match.group(2)) if ml_match else 0,
                    'gpt_recommendation': gpt_match.group(1) if gpt_match else 'UNKNOWN',
                    'reliability_score': float(reliability_match.group(1)) if reliability_match else 0
                }
                data['texts'].append(text_data)
            except Exception as e:
                print(f"Erro ao processar texto {text_num}: {e}")
                continue
                
    except Exception as e:
        print(f"Erro no parsing do relatório: {e}")
        raise HTTPException(status_code=500, detail="Erro ao processar relatório")
    
    return data

def generate_dashboard_image(data: Dict, theme: str) -> BytesIO:
    """Gera imagem PNG do dashboard"""
    plt.style.use('default')
    sns.set_palette("Set2")
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'Dashboard de Análise - Tema: {theme}', fontsize=16, fontweight='bold')
    
    # Layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
    
    # 1. Gráfico de pizza
    ax1 = fig.add_subplot(gs[0, 0])
    labels = ['FAKE', 'TRUE']
    sizes = [data['summary']['fake_percentage'], data['summary']['true_percentage']]
    colors = ['#ff6b6b', '#51cf66']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribuição FAKE vs TRUE')
    ax1.axis('equal')
    
    # 2. Métricas gerais
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['Confiança Média', 'Score Confiabilidade']
    values = [data['summary']['avg_confidence'], data['summary']['avg_reliability']]
    bars = ax2.bar(categories, values, color=['#339af0', '#ff922b'])
    ax2.set_ylabel('Percentual (%)')
    ax2.set_ylim(0, 100)
    ax2.set_title('Métricas Gerais de Confiança')
    
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 3. Comparação por texto
    ax3 = fig.add_subplot(gs[1, :])
    texts = data['texts']
    text_numbers = [f'Texto {t["number"]}' for t in texts]
    ml_confidences = [t["ml_confidence"] for t in texts]
    reliability_scores = [t["reliability_score"] for t in texts]
    
    x = np.arange(len(text_numbers))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, ml_confidences, width, label='Confiança ML', color='#339af0')
    bars2 = ax3.bar(x + width/2, reliability_scores, width, label='Score Confiabilidade', color='#ff922b')
    
    ax3.set_xlabel('Textos Analisados')
    ax3.set_ylabel('Scores (%)')
    ax3.set_title('Comparação por Texto: ML vs Confiabilidade')
    ax3.set_xticks(x)
    ax3.set_xticklabels(text_numbers)
    ax3.legend()
    ax3.set_ylim(0, 100)
    
    # 4. Tabela de resumo
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis('off')
    
    summary_data = [
        ['Total de Textos', data['summary']['total_texts']],
        ['Textos FAKE', f"{data['summary']['fake_count']} ({data['summary']['fake_percentage']}%)"],
        ['Textos TRUE', f"{data['summary']['true_count']} ({data['summary']['true_percentage']}%)"],
        ['Confiança Média', f"{data['summary']['avg_confidence']}%"],
        ['Score Confiabilidade Médio', f"{data['summary']['avg_reliability']}%"]
    ]
    
    table = ax4.table(cellText=summary_data,
                     cellLoc='left',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    
    buf.seek(0)
    return buf

async def analyze_theme_with_gpt(theme: str, max_posts: int = 10, language: str = 'pt') -> Dict:
    """Executa análise do tema usando o modelo GPT"""
    # Simulação de coleta de posts (substitua pela sua implementação real)
    print(f"Analisando tema: {theme}, posts: {max_posts}, idioma: {language}")
    
    # Chama sua função existente (ajuste conforme necessário)
    #report_filename = test_enhanced_model(openai_api_key=API_KEY)
    report_filename = "respostas.txt"
    
    # Lê e parseia o relatório
    with open(report_filename, 'r', encoding='utf-8') as f:
        report_content = f.read()
    
    return parse_report(report_content)

@app.get("/")
async def root():
    """Endpoint raiz com informações da API unificada"""
    return {
        "message": "Unified Fake News Detection API",
        "version": "2.0.0",
        "services": {
            "bluesky_analysis": "Análise de posts do Bluesky por tema",
            "enhanced_model": "Modelo aprimorado de detecção com GPT-4"
        },
        "endpoints": {
            "health": "/health",
            "bluesky_analyze": "/analyze-theme",
            "bluesky_dashboard": "/dashboard/{theme}",
            "bluesky_posts": "/posts/{theme}",
            "enhanced_predict": "/predict",
            "enhanced_report": "/predict/report/{filename}",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/analyze-theme")
async def analyze_theme(request: ThemeAnalysisRequest):
    """Analisa um tema e retorna dados estruturados"""
    try:
        # Verifica se já existe no cache
        cache_key = f"{request.theme}_{request.max_posts}_{request.language}"
        if cache_key in analysis_cache:
            return analysis_cache[cache_key]
        
        # Executa análise
        analysis_data = await analyze_theme_with_gpt(request.theme, request.max_posts, request.language)
        
        # Armazena no cache
        analysis_cache[cache_key] = analysis_data
        
        return analysis_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

@app.get("/dashboard/{theme}")
async def get_dashboard(theme: str, max_posts: int = 10):
    """Retorna dashboard como imagem PNG"""
    try:
        cache_key = f"{theme}_{max_posts}_pt"
        
        if cache_key not in analysis_cache:
            # Se não existe, executa análise primeiro
            analysis_data = await analyze_theme_with_gpt(theme, max_posts)
            analysis_cache[cache_key] = analysis_data
        
        data = analysis_cache[cache_key]
        image_buf = generate_dashboard_image(data, theme)
        
        return Response(
            content=image_buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f"inline; filename=dashboard_{theme}.png"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar dashboard: {str(e)}")

@app.get("/posts/{theme}")
async def get_posts(theme: str, max_posts: int = 10):
    """Retorna os posts analisados (dados brutos)"""
    try:
        cache_key = f"{theme}_{max_posts}_pt"
        
        if cache_key not in analysis_cache:
            analysis_data = await analyze_theme_with_gpt(theme, max_posts)
            analysis_cache[cache_key] = analysis_data
        
        # Retorna os textos analisados
        posts_data = []
        for text in analysis_cache[cache_key]['texts']:
            posts_data.append({
                'id': text['number'],
                'ml_prediction': text['ml_prediction'],
                'ml_confidence': text['ml_confidence'],
                'gpt_recommendation': text['gpt_recommendation'],
                'reliability_score': text['reliability_score']
            })
        
        return {
            'theme': theme,
            'total_posts': analysis_cache[cache_key]['summary']['total_texts'],
            'posts': posts_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter posts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)