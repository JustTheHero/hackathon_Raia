import { useState } from 'react'
import Header from './components/Header'
import SearchForm from './components/SearchForm'
import PostList from './components/PostList'
import Dashboard from './components/Dashboard'
import Loading from './components/Loading'
import { analyzeTheme } from './services/api'
import './App.css'

function App() {
  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (theme, maxPosts) => {
    try {
      setLoading(true)
      setError(null)
      const data = await analyzeTheme(theme, maxPosts)
      setResults(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao processar a requisição')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="App">
      <Header />
      
      <main className="main-content">
        <SearchForm onSearch={handleSearch} loading={loading} />
        
        {loading && <Loading />}
        
        {error && (
          <div className="error-message">
            <h3>Erro</h3>
            <p>{error}</p>
          </div>
        )}
        
        {results && !loading && (
          <div className="results">
            <div className="summary">
              <h2>Resultados para: {results.theme}</h2>
              <p>Total de posts analisados: {results.total_posts}</p>
              <p>Fake news detectadas: {results.fake_news_count} ({results.analysis_summary.fake_news_percentage}%)</p>
            </div>
            
            <Dashboard theme={results.theme} maxPosts={results.total_posts} />
            
            <PostList posts={results.posts} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App