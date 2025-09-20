import React from 'react';
import './PostList.css';

const PostList = ({ posts }) => {
  if (!posts || posts.length === 0) {
    return <div className="no-posts">Nenhum post encontrado.</div>;
  }

  return (
    <div className="post-list">
      <h2>Posts Analisados</h2>
      <div className="posts-container">
        {posts.map((post, index) => (
          <div key={index} className={`post-card ${post.is_fake_news ? 'fake-news' : 'true-news'}`}>
            <div className="post-header">
              <span className="author">@{post.author}</span>
              <span className="date">{new Date(post.createdAt).toLocaleDateString()}</span>
            </div>
            
            <div className="post-content">
              <p>{post.text}</p>
            </div>
            
            <div className="post-stats">
              <div className="engagement">
                <span>❤️ {post.likes}</span>
                <span>💬 {post.replies}</span>
                <span>🔁 {post.reposts}</span>
              </div>
              
              <div className="analysis">
                <div className={`fake-score ${post.is_fake_news ? 'high' : 'low'}`}>
                  Score Fake News: {post.fake_news_score}
                </div>
                <div className="confidence">
                  Confiança: {(post.confidence * 100).toFixed(1)}%
                </div>
                <div className={`verdict ${post.is_fake_news ? 'fake' : 'true'}`}>
                  {post.is_fake_news ? '⚠️ POTENCIAL FAKE NEWS' : '✅ CONTEÚDO VERÍDICO'}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default PostList;