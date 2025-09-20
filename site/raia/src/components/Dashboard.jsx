import React, { useState, useEffect } from 'react';
import { getDashboard } from '../services/api';
import './Dashboard.css';

const Dashboard = ({ theme, maxPosts }) => {
  const [dashboardUrl, setDashboardUrl] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchDashboard = async () => {
      try {
        setLoading(true);
        const url = await getDashboard(theme, maxPosts);
        setDashboardUrl(url);
        setError(null);
      } catch (err) {
        setError('Erro ao carregar dashboard');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    if (theme) {
      fetchDashboard();
    }
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
        <img src={dashboardUrl} alt={`Dashboard de análise para ${theme}`} />
      </div>
    </div>
  );
};

export default Dashboard;