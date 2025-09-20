import React, { useState } from 'react';
import './SearchForm.css';

const SearchForm = ({ onSearch, loading }) => {
  const [theme, setTheme] = useState('');
  const [maxPosts, setMaxPosts] = useState(10);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (theme.trim()) {
      onSearch(theme, maxPosts);
    }
  };

  return (
    <form className="search-form" onSubmit={handleSubmit}>
      <div className="form-group">
        <label htmlFor="theme">Tema para pesquisa:</label>
        <input
          type="text"
          id="theme"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          placeholder="Digite um tema para analisar (ex: vacinas, política, etc.)"
          disabled={loading}
          required
        />
      </div>
      
      <div className="form-group">
        <label htmlFor="maxPosts">Número máximo de posts:</label>
        <input
          type="number"
          id="maxPosts"
          value={maxPosts}
          onChange={(e) => setMaxPosts(parseInt(e.target.value) || 10)}
          min="1"
          max="50"
          disabled={loading}
        />
      </div>
      
      <button type="submit" disabled={loading || !theme.trim()}>
        {loading ? 'Analisando...' : 'Analisar Tema'}
      </button>
    </form>
  );
};

export default SearchForm;