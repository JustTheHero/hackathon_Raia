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
  const [theme, setTheme] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleSearch = async (searchTheme, maxPosts) => {
    try {
      setLoading(true)
      setError(null)
      setTheme(searchTheme)
      const data = await analyzeTheme(searchTheme, maxPosts)
      setResults(data)
    } catch (err) {
      setError(err.response?.data?.detail || 'Erro ao processar a requisição')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // Adicione esta verificação para evitar erros
  if (results && !results.summary) {
    console.error('Estrutura inesperada dos resultados:', results)
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
        
        {results && results.summary && !loading && (
          <div className="results">
            <div className="summary">
              <h2>Resultados para: {theme}</h2>
              <p>Total de textos analisados: {results.summary.total_texts}</p>
              <p>Classificados como FAKE: {results.summary.fake_count} ({results.summary.fake_percentage}%)</p>
              <p>Classificados como TRUE: {results.summary.true_count} ({results.summary.true_percentage}%)</p>
              <p>Confiança média: {results.summary.avg_confidence}%</p>
              <p>Score de confiabilidade médio: {results.summary.avg_reliability}%</p>
            </div>
            
            <Dashboard 
              theme={theme} 
              data={results}
              maxPosts={results.summary.total_texts} 
            />
            
            <PostList posts={results.texts || []} />
          </div>
        )}

        {/* Adicione esta verificação para resultados inesperados */}
        {results && !results.summary && !loading && (
          <div className="error-message">
            <h3>Formato de dados inesperado</h3>
            <p>Os resultados retornaram em um formato diferente do esperado.</p>
            <pre>{JSON.stringify(results, null, 2)}</pre>
          </div>
        )}
      </main>
    </div>
  )
}

export default App