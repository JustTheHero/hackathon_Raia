import React, { useState, useEffect } from 'react';
import { getDashboard } from '../services/api';
import './Dashboard.css';

const Dashboard = ({ theme, maxPosts }) => {
  const [dashboardImage, setDashboardImage] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setLoading(true);
        setError(null);
        
        // A API retorna a imagem diretamente, não uma URL
        const imageBlob = await getDashboard(theme, maxPosts);
        
        // Cria uma URL para o blob da imagem
        const imageUrl = URL.createObjectURL(imageBlob);
        setDashboardImage(imageUrl);
        
      } catch (err) {
        setError('Erro ao carregar dashboard');
        console.error('Erro no dashboard:', err);
      } finally {
        setLoading(false);
      }
    };

    if (theme) {
      fetchDashboard();
    }

    // Cleanup function para revogar a URL quando o componente desmontar
    return () => {
      if (dashboardImage) {
        URL.revokeObjectURL(dashboardImage);
      }
    };
  }, [theme, maxPosts]);

  if (loading) {
    return <div className="dashboard-loading">Carregando dashboard...</div>;
  }

  if (error) {
    return <div className="dashboard-error">{error}</div>;
  }

  return (
    <div className="dashboard">
      <h2>Dashboard de Análise - {theme}</h2>
      <div className="dashboard-image">
        {dashboardImage && (
          <img 
            src={dashboardImage} 
            alt={`Dashboard de análise para ${theme}`}
            onError={(e) => {
              console.error('Erro ao carregar imagem do dashboard');
              e.target.style.display = 'none';
            }}
          />
        )}
      </div>
    </div>
  );
};

export default Dashboard;