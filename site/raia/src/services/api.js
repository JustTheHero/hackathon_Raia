import axios from 'axios';

// Configuração do axios para usar o proxy do Vite
const API_BASE_URL = '/api';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const analyzeTheme = async (theme, maxPosts = 10, language = 'pt') => {
  try {
    const response = await api.post('/analyze-theme', {
      theme,
      max_posts: maxPosts,
      language,
    });
    return response.data;
  } catch (error) {
    console.error('Erro ao analisar tema:', error);
    throw error;
  }
};

export const getDashboard = async (theme, maxPosts = 10) => {
  try {
    const response = await api.get(`/dashboard/${theme}?max_posts=${maxPosts}`, {
      responseType: 'blob',
    });
    return URL.createObjectURL(response.data);
  } catch (error) {
    console.error('Erro ao obter dashboard:', error);
    throw error;
  }
};

export const getPosts = async (theme, maxPosts = 10) => {
  try {
    const response = await api.get(`/posts/${theme}?max_posts=${maxPosts}`);
    return response.data;
  } catch (error) {
    console.error('Erro ao obter posts:', error);
    throw error;
  }
};

export const healthCheck = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    console.error('Erro no health check:', error);
    throw error;
  }
};

export default api;