from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import io
import base64
import asyncio
import logging
import tempfile
import os
from models.model import test_enhanced_model

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Fake News Detection API",
    description="API para buscar posts do Bluesky, detectar fake news e gerar dashboards visuais",
    version="1.0.0"
)

API_KEY = ""

class Post(BaseModel):
    """Modelo para representar um post"""
    uri: str
    text: str
    author: str
    createdAt: str
    likes: int
    replies: int
    reposts: int
    fake_news_score: Optional[float] = None
    is_fake_news: Optional[bool] = None
    confidence: Optional[float] = None

class ThemeRequest(BaseModel):
    """Modelo para requisição de tema"""
    theme: str
    max_posts: Optional[int] = 10
    language: Optional[str] = "pt"

class AnalysisResponse(BaseModel):
    """Modelo para resposta da análise"""
    theme: str
    total_posts: int
    fake_news_count: int
    posts: List[Post]
    analysis_summary: dict
    dashboard_image: Optional[str] = None  # Base64 encoded image

def generate_dashboard(analysis_data: dict, posts: List[Post]) -> str:
    """
    Gera um dashboard visual com as estatísticas de fake news
    Retorna a imagem em base64
    """
    try:
        # Configurar o estilo do matplotlib
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Dashboard de Análise de Fake News - Tema: {analysis_data["theme"]}', 
                    fontsize=16, fontweight='bold')
        
        # Dados para os gráficos
        labels = ['Verdadeiros', 'Fake News']
        sizes = [analysis_data['true_news_count'], analysis_data['fake_news_count']]
        colors = ['#66b3ff', '#ff6666']
        
        # Gráfico 1: Pizza - Distribuição de Fake News
        axes[0, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Distribuição de Fake News vs. Conteúdo Verdadeiro')
        axes[0, 0].axis('equal')
        
        # Gráfico 2: Barras - Engajamento por tipo de conteúdo
        fake_news_posts = [p for p in posts if p.is_fake_news]
        true_news_posts = [p for p in posts if not p.is_fake_news]
        
        fake_engagement = sum([p.likes + p.replies * 2 + p.reposts * 3 for p in fake_news_posts]) / len(fake_news_posts) if fake_news_posts else 0
        true_engagement = sum([p.likes + p.replies * 2 + p.reposts * 3 for p in true_news_posts]) / len(true_news_posts) if true_news_posts else 0
        
        engagement_data = [true_engagement, fake_engagement]
        bars = axes[0, 1].bar(labels, engagement_data, color=colors)
        axes[0, 1].set_title('Engajamento Médio por Tipo de Conteúdo')
        axes[0, 1].set_ylabel('Pontuação de Engajamento')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, engagement_data):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # Gráfico 3: Distribuição de scores de fake news
        scores = [p.fake_news_score for p in posts]
        axes[1, 0].hist(scores, bins=10, color='#ff9999', edgecolor='black')
        axes[1, 0].set_title('Distribuição de Scores de Fake News')
        axes[1, 0].set_xlabel('Score de Fake News')
        axes[1, 0].set_ylabel('Frequência')
        axes[1, 0].axvline(x=0.5, color='red', linestyle='--', label='Limite Fake News')
        axes[1, 0].legend()
        
        # Gráfico 4: Top autores com mais fake news
        fake_authors = {}
        for post in fake_news_posts:
            fake_authors[post.author] = fake_authors.get(post.author, 0) + 1
        
        if fake_authors:
            author_names = list(fake_authors.keys())
            author_counts = list(fake_authors.values())
            
            # Ordenar e pegar os top 5
            sorted_indices = np.argsort(author_counts)[::-1][:5]
            top_authors = [author_names[i] for i in sorted_indices]
            top_counts = [author_counts[i] for i in sorted_indices]
            
            bars = axes[1, 1].barh(top_authors, top_counts, color='#ff6666')
            axes[1, 1].set_title('Top Autores com Fake News')
            axes[1, 1].set_xlabel('Número de Fake News')
            
            # Adicionar valores nas barras
            for i, (bar, value) in enumerate(zip(bars, top_counts)):
                axes[1, 1].text(value + 0.1, bar.get_y() + bar.get_height()/2,
                               f'{value}', ha='left', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'Nenhuma fake news detectada', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Top Autores com Fake News')
        
        plt.tight_layout()
        
        # Salvar a imagem em um buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Codificar a imagem em base64
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)  # Fechar a figura para liberar memória
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Erro ao gerar dashboard: {str(e)}")
        # Gerar uma imagem de erro
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Erro ao gerar dashboard: {str(e)}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Erro na Geração do Dashboard")
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64

@app.get("/")
async def root():
    """Endpoint raiz"""
    return {
        "message": "Fake News Detection API",
        "endpoints": {
            "health": "/health",
            "dashboard": "/dashboard/{theme}",
        }
    }

@app.get("/health")
async def health_check():
    """Endpoint de health check"""
    return {"status": "healthy", "service": "fake-news-detection"}

@app.get("/dashboard/{theme}")
async def get_dashboard(theme: str, max_posts: int = 10):
    """
    Endpoint para retornar apenas o dashboard visual
    """
    try:
        # Buscar e analisar posts
        posts = await fetch_bluesky_posts(theme, max_posts)
        analyzed_posts = await analyze_fake_news(posts)
        
        # Calcular estatísticas
        fake_news_count = sum(1 for post in analyzed_posts if post.is_fake_news)
        true_news_count = len(analyzed_posts) - fake_news_count
        
        # Gerar dashboard
        analysis_data = {
            "theme": theme,
            "total_posts": len(analyzed_posts),
            "fake_news_count": fake_news_count,
            "true_news_count": true_news_count,
        }
        
        dashboard_image = generate_dashboard(analysis_data, analyzed_posts)
        
        # Decodificar a imagem base64 e retornar como PNG
        image_data = base64.b64decode(dashboard_image)
        return Response(content=image_data, media_type="image/png")
        
    except Exception as e:
        logger.error(f"Erro ao gerar dashboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar dashboard: {str(e)}")

@app.get("/posts/{theme}")
async def get_posts_only(theme: str, max_posts: int = 10):
    """
    Endpoint alternativo para apenas buscar posts (sem análise)
    """
    try:
        posts = await fetch_bluesky_posts(theme, max_posts)
        return {"theme": theme, "posts": posts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)