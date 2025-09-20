import React, { useState } from 'react';
import './PostList.css';

const PostList = ({ posts }) => {
  const [expandedPosts, setExpandedPosts] = useState({});

  if (!posts || posts.length === 0) {
    return (
      <div className="post-list">
        <h3>Nenhum texto analisado</h3>
        <p>Não foram encontrados textos para exibir.</p>
      </div>
    );
  }

  const toggleExpand = (postNumber) => {
    setExpandedPosts(prev => ({
      ...prev,
      [postNumber]: !prev[postNumber]
    }));
  };

  const getPredictionColor = (prediction) => {
    switch (prediction?.toUpperCase()) {
      case 'FAKE':
        return '#dc3545';
      case 'TRUE':
        return '#28a745';
      default:
        return '#6c757d';
    }
  };

  const getReliabilityColor = (score) => {
    if (score >= 80) return '#28a745';
    if (score >= 60) return '#ffc107';
    return '#dc3545';
  };

  // Função para extrair apenas o conteúdo do preview (remover "Preview: ")
  const getCleanPreview = (preview) => {
    if (!preview) return 'Conteúdo não disponível'; // Está com problema, era pra ser o preview do texto
    
    // Remove o "Preview: " se existir
    const cleanPreview = preview.replace(/^Preview:\s*/i, '').trim();
    
    // Remove espaços em excesso e quebras de linha no início
    return cleanPreview.replace(/^\s+/, '');
  };

  return (
    <div className="post-list">
      <h3>Análise Detalhada dos Textos</h3>
      <p className="post-list-subtitle">
        {posts.length} texto(s) analisado(s) com modelo ML e GPT-4
      </p>
      
      <div className="posts-grid">
        {posts.map((post) => {
          const cleanPreview = getCleanPreview(post.preview);
          
          return (
            <div 
              key={post.number} 
              className="post-card"
              style={{ 
                borderLeft: `4px solid ${getPredictionColor(post.ml_prediction)}` 
              }}
            >
              <div className="post-header">
                <h4>📝 Texto #{post.number}</h4>
                <span 
                  className="prediction-badge"
                  style={{ backgroundColor: getPredictionColor(post.ml_prediction) }}
                >
                  {post.ml_prediction || 'INDEFINIDO'}
                </span>
              </div>

              {/* Conteúdo do preview como título principal */}
              <div className="preview-content-main">
                {expandedPosts[post.number] 
                  ? cleanPreview
                  : `${cleanPreview.substring(0, 120)}${cleanPreview.length > 120 ? '...' : ''}`
                }
              </div>

              {cleanPreview.length > 120 && (
                <button 
                  className="expand-button"
                  onClick={() => toggleExpand(post.number)}
                >
                  {expandedPosts[post.number] ? 'Ver menos' : 'Ver mais'}
                </button>
              )}

              <div className="post-details">
                <div className="detail-item">
                  <span className="detail-label">🧠 Predição ML:</span>
                  <span className="detail-value">
                    {post.ml_prediction || 'N/A'} 
                    {post.ml_confidence && ` (${post.ml_confidence}% confiança)`}
                  </span>
                </div>

                <div className="detail-item">
                  <span className="detail-label">🤖 Recomendação GPT:</span>
                  <span className="detail-value">
                    {post.gpt_recommendation || 'N/A'}
                  </span>
                </div>

                <div className="detail-item">
                  <span className="detail-label">⭐ Score de Confiabilidade:</span>
                  <span 
                    className="detail-value reliability-score"
                    style={{ color: getReliabilityColor(post.reliability_score) }}
                  >
                    {post.reliability_score}%
                  </span>
                </div>
              </div>

              {(post.ml_prediction && post.gpt_recommendation) && (
                <div className="consistency-status">
                  {post.ml_prediction.toUpperCase() === post.gpt_recommendation.toUpperCase() ? (
                    <span className="consistent">✅ Análises consistentes</span>
                  ) : (
                    <span className="inconsistent">⚠️ Análises divergentes</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PostList;