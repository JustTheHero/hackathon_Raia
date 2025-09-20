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
from models.isso import test_enhanced_model

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Unified Fake News Detection API",
    description="API unificada para detecção de fake news - Bluesky + Modelo Aprimorado",
    version="2.0.0"
)

# API Key para OpenAI (configure como variável de ambiente em produção)
API_KEY = ""  # Configure sua API key aqui ou use variáveis de ambiente

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

class EnhancedModelResponse(BaseModel):
    """Modelo para resposta do modelo aprimorado"""
    status: str
    report_filename: str
    summary: dict
    message: str

async def fetch_bluesky_posts(theme: str, max_posts: int = 10) -> List[Post]:
    """
    Simula a busca de posts no Bluesky baseado no tema
    Em uma implementação real, usaria a API oficial do Bluesky
    """
    logger.info(f"Buscando posts sobre: {theme}")
    
    # Simulação de resposta da API do Bluesky com dados mais realistas
    mock_posts_data = [
        {
            "text": f"Atenção: novo golpe está circulando sobre {theme}. Não cliquem em links suspeitos!",
            "author": "SegurançaOnline",
            "likes": 45,
            "replies": 12,
            "reposts": 8
        },
        {
            "text": f"URGENTE: Descoberta revolucionária sobre {theme} que vai mudar tudo!",
            "author": "CuriosoNews",
            "likes": 120,
            "replies": 25,
            "reposts": 15
        },
        {
            "text": f"Especialistas confirmam: {theme} é uma das maiores ameaças atuais",
            "author": "CiênciaHoje",
            "likes": 89,
            "replies": 15,
            "reposts": 10
        },
        {
            "text": f"ALERTA: Governo está escondendo a verdade sobre {theme}",
            "author": "VerdadeOculta",
            "likes": 230,
            "replies": 45,
            "reposts": 32
        },
        {
            "text": f"Estudo comprova: {theme} não é tão perigoso quanto dizem",
            "author": "FatosCientíficos",
            "likes": 76,
            "replies": 18,
            "reposts": 7
        },
        {
            "text": f"Pesquisa da universidade revela novos dados sobre {theme}",
            "author": "UniversidadeNews",
            "likes": 65,
            "replies": 12,
            "reposts": 5
        },
        {
            "text": f"Ministério da Saúde se pronuncia sobre {theme}",
            "author": "GovOficial",
            "likes": 134,
            "replies": 28,
            "reposts": 19
        },
        {
            "text": f"CUIDADO! Nova variante de {theme} detectada!!!",
            "author": "AlertaUrgente",
            "likes": 87,
            "replies": 33,
            "reposts": 24
        },
        {
            "text": f"Organização Mundial da Saúde atualiza diretrizes sobre {theme}",
            "author": "SaúdeMundial",
            "likes": 156,
            "replies": 41,
            "reposts": 22
        },
        {
            "text": f"Cientistas descobrem ligação entre {theme} e outros problemas de saúde",
            "author": "CiênciaAtual",
            "likes": 92,
            "replies": 17,
            "reposts": 11
        }
    ]
    
    # Garantir que não excedemos o max_posts
    posts_data = mock_posts_data[:max_posts]
    
    posts = []
    for i, data in enumerate(posts_data):
        post = Post(
            uri=f"at://did:plc:example/post_{i}",
            text=data["text"],
            author=data["author"],
            createdAt="2024-01-01T12:00:00Z",
            likes=data["likes"],
            replies=data["replies"],
            reposts=data["reposts"]
        )
        posts.append(post)
    
    await asyncio.sleep(1)  # Simula delay de rede
    return posts

async def analyze_fake_news(posts: List[Post]) -> List[Post]:
    """
    Simula a análise de fake news usando um modelo ML
    Em produção, integraria com o modelo real
    """
    logger.info("Analisando posts para detecção de fake news")
    
    analyzed_posts = []
    for i, post in enumerate(posts):
        # Simulação de scores do modelo de fake news com base no conteúdo do post
        text = post.text.lower()
        
        # Heurísticas simples para demonstração
        if "urgente" in text or "alerta" in text or "revolucionária" in text or "cuidado" in text:
            fake_news_score = round(0.7 + (np.random.random() * 0.2), 2)
        elif "estudo" in text or "comprova" in text or "especialistas" in text or "pesquisa" in text:
            fake_news_score = round(0.3 + (np.random.random() * 0.3), 2)
        elif "ministério" in text or "organização mundial" in text or "gov" in text:
            fake_news_score = round(0.2 + (np.random.random() * 0.3), 2)
        else:
            fake_news_score = round(0.1 + (np.random.random() * 0.4), 2)
            
        is_fake_news = fake_news_score > 0.5
        confidence = round(np.random.random() * 0.3 + 0.7, 2)  # Confiança entre 0.7-1.0
        
        analyzed_post = post.copy()
        analyzed_post.fake_news_score = fake_news_score
        analyzed_post.is_fake_news = is_fake_news
        analyzed_post.confidence = confidence
        analyzed_posts.append(analyzed_post)
    
    await asyncio.sleep(0.5)  # Simula processamento do modelo
    return analyzed_posts

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

def parse_enhanced_model_report(report_content: str) -> dict:
    """
    Extrai informações do relatório do modelo aprimorado
    """
    try:
        lines = report_content.split('\n')
        
        # Extrair estatísticas gerais
        summary = {
            "total_texts": 0,
            "fake_count": 0,
            "true_count": 0,
            "avg_confidence": 0.0,
            "avg_reliability": 0.0,
            "texts_analyzed": []
        }
        
        for line in lines:
            if "Total de textos analisados:" in line:
                summary["total_texts"] = int(line.split(":")[1].strip())
            elif "Classificados como FAKE:" in line:
                fake_info = line.split(":")[1].strip()
                summary["fake_count"] = int(fake_info.split()[0])
            elif "Classificados como TRUE:" in line:
                true_info = line.split(":")[1].strip()
                summary["true_count"] = int(true_info.split()[0])
            elif "Confiança média:" in line:
                summary["avg_confidence"] = float(line.split(":")[1].strip().replace('%', ''))
            elif "Score de confiabilidade médio:" in line:
                summary["avg_reliability"] = float(line.split(":")[1].strip().replace('%', ''))
        
        return summary
        
    except Exception as e:
        logger.error(f"Erro ao analisar relatório: {str(e)}")
        return {"error": str(e)}

# ============================================================================
# ENDPOINTS ORIGINAIS DA API DE FAKE NEWS (BLUESKY)
# ============================================================================

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
    """Endpoint de health check"""
    return {
        "status": "healthy", 
        "services": {
            "bluesky_service": "operational",
            "enhanced_model": "operational" if API_KEY else "api_key_required"
        }
    }

@app.post("/analyze-theme", response_model=AnalysisResponse)
async def analyze_theme(request: ThemeRequest):
    """
    Endpoint principal para analisar posts de um tema específico no Bluesky
    """
    try:
        # 1. Buscar posts no Bluesky
        posts = await fetch_bluesky_posts(
            theme=request.theme,
            max_posts=request.max_posts
        )
        
        if not posts:
            raise HTTPException(
                status_code=404,
                detail=f"Nenhum post encontrado para o tema: {request.theme}"
            )
        
        # 2. Analisar posts com modelo de fake news
        analyzed_posts = await analyze_fake_news(posts)
        
        # 3. Calcular estatísticas
        fake_news_count = sum(1 for post in analyzed_posts if post.is_fake_news)
        true_news_count = len(analyzed_posts) - fake_news_count
        avg_score = round(
            sum(post.fake_news_score for post in analyzed_posts) / len(analyzed_posts), 
            2
        ) if analyzed_posts else 0
        
        # 4. Gerar dashboard
        analysis_data = {
            "theme": request.theme,
            "total_posts": len(analyzed_posts),
            "fake_news_count": fake_news_count,
            "true_news_count": true_news_count,
            "fake_news_percentage": round((fake_news_count / len(analyzed_posts)) * 100, 2),
            "average_fake_score": avg_score,
        }
        
        dashboard_image = generate_dashboard(analysis_data, analyzed_posts)
        
        # 5. Preparar resposta
        response = AnalysisResponse(
            theme=request.theme,
            total_posts=len(analyzed_posts),
            fake_news_count=fake_news_count,
            posts=analyzed_posts,
            analysis_summary=analysis_data,
            dashboard_image=dashboard_image
        )
        
        logger.info(f"Análise concluída para tema: {request.theme}")
        return response
        
    except Exception as e:
        logger.error(f"Erro ao processar requisição: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

@app.get("/dashboard/{theme}")
async def get_dashboard(theme: str, max_posts: int = 10):
    """
    Endpoint para retornar apenas o dashboard visual do Bluesky
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

# ============================================================================
# NOVOS ENDPOINTS DO MODELO APRIMORADO
# ============================================================================

@app.get("/predict", response_model=EnhancedModelResponse)
def run_prediction():
    """
    Executa o modelo aprimorado de detecção de fake news com GPT-4
    """
    try:
        if not API_KEY:
            raise HTTPException(
                status_code=400, 
                detail="API Key do OpenAI não configurada. Configure a variável API_KEY."
            )
        
        logger.info("Executando modelo aprimorado de detecção de fake news")
        
        # Executar o modelo aprimorado
        report_filename = test_enhanced_model(openai_api_key=API_KEY)
        
        # Ler e analisar o relatório
        try:
            with open(report_filename, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            summary = parse_enhanced_model_report(report_content)
            
            response = EnhancedModelResponse(
                status="success",
                report_filename=report_filename,
                summary=summary,
                message=f"Análise concluída. Relatório salvo em: {report_filename}"
            )
            
            logger.info(f"Modelo aprimorado executado com sucesso. Relatório: {report_filename}")
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar relatório: {str(e)}")
            return EnhancedModelResponse(
                status="partial_success",
                report_filename=report_filename,
                summary={"error": f"Relatório gerado mas erro ao processar: {str(e)}"},
                message=f"Modelo executado mas erro ao processar relatório: {str(e)}"
            )
        
    except Exception as e:
        logger.error(f"Erro ao executar modelo aprimorado: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao executar modelo: {str(e)}")

@app.get("/predict/download")
def download_latest_report():
    """
    Retorna o último relatório gerado como download
    """
    try:
        # Executar o modelo para gerar um novo relatório
        if not API_KEY:
            raise HTTPException(
                status_code=400, 
                detail="API Key do OpenAI não configurada"
            )
            
        report_filename = test_enhanced_model(openai_api_key=API_KEY)
        
        if not os.path.exists(report_filename):
            raise HTTPException(
                status_code=404, 
                detail="Arquivo de relatório não encontrado"
            )
        
        return FileResponse(
            report_filename,
            media_type="text/plain",
            filename="fake_news_report.txt",
            headers={"Content-Disposition": "attachment; filename=fake_news_report.txt"}
        )
        
    except Exception as e:
        logger.error(f"Erro ao baixar relatório: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar/baixar relatório: {str(e)}")

@app.get("/predict/report/{filename}")
def get_report_file(filename: str):
    """
    Retorna um relatório específico pelo nome do arquivo
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(filename):
            raise HTTPException(
                status_code=404, 
                detail=f"Relatório não encontrado: {filename}"
            )
        
        return FileResponse(
            filename,
            media_type="text/plain",
            filename=os.path.basename(filename)
        )
        
    except Exception as e:
        logger.error(f"Erro ao acessar relatório: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro ao acessar relatório: {str(e)}")

@app.get("/predict/status")
def get_prediction_status():
    """
    Retorna o status do serviço de predição aprimorada
    """
    return {
        
        "service": "Enhanced Fake News Detection",
        "api_key_configured": bool(API_KEY),
        "model": "GPT-4 Mini",
        "status": "ready" if API_KEY else "api_key_required",
        "endpoints": {
            "predict": "/predict",
            "download": "/predict/download", 
            "status": "/predict/status"
        }
    }

# ============================================================================
# CONFIGURAÇÃO PARA EXECUÇÃO
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Configurar API_KEY se não estiver definida
    if not API_KEY:
        logger.warning("API_KEY não configurada. O serviço de modelo aprimorado não funcionará.")
        logger.info("Configure a API_KEY do OpenAI para usar o modelo aprimorado.")
    
    logger.info("Iniciando API Unificada de Detecção de Fake News...")
    logger.info("Serviços disponíveis:")
    logger.info("- Análise de posts do Bluesky: /analyze-theme")
    logger.info("- Modelo aprimorado GPT-4: /predict")
    logger.info("- Documentação: /docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
