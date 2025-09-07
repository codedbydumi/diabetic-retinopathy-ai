import React, { useState, useEffect } from 'react';
import { 
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, PieChart, Pie, Cell
} from 'recharts';

const Dashboard = () => {
  const [stats, setStats] = useState({
    totalPredictions: 1248,
    accuracy: 92.4,
    highRiskCases: 312,
    avgResponseTime: 183
  });

  const riskDistribution = [
    { name: 'Low', value: 45, color: '#00A651' },
    { name: 'Medium', value: 30, color: '#FFB800' },
    { name: 'High', value: 20, color: '#DC3545' },
    { name: 'Very High', value: 5, color: '#8B0000' }
  ];

  const monthlyTrends = [
    { month: 'Jan', predictions: 180, accuracy: 88 },
    { month: 'Feb', predictions: 195, accuracy: 89 },
    { month: 'Mar', predictions: 210, accuracy: 90 },
    { month: 'Apr', predictions: 225, accuracy: 91 },
    { month: 'May', predictions: 238, accuracy: 91.5 },
    { month: 'Jun', predictions: 200, accuracy: 92.4 }
  ];

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h2>System Dashboard</h2>
        <p>Real-time monitoring and analytics</p>
      </div>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon">📊</div>
          <div className="stat-content">
            <h3>Total Predictions</h3>
            <p className="stat-value">{stats.totalPredictions.toLocaleString()}</p>
            <span className="stat-change positive">+12% this month</span>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">🎯</div>
          <div className="stat-content">
            <h3>Model Accuracy</h3>
            <p className="stat-value">{stats.accuracy}%</p>
            <span className="stat-change positive">+2.1% improvement</span>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">⚠️</div>
          <div className="stat-content">
            <h3>High Risk Cases</h3>
            <p className="stat-value">{stats.highRiskCases}</p>
            <span className="stat-change">25% of total</span>
          </div>
        </div>
        
        <div className="stat-card">
          <div className="stat-icon">⏱️</div>
          <div className="stat-content">
            <h3>Avg Response</h3>
            <p className="stat-value">{stats.avgResponseTime}ms</p>
            <span className="stat-change positive">-20ms faster</span>
          </div>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-card">
          <h3>Risk Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={riskDistribution}
                cx="50%"
                cy="50%"
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
                label={({name, value}) => `${name}: ${value}%`}
              >
                {riskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>Monthly Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={monthlyTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Bar yAxisId="left" dataKey="predictions" fill="#E8F4F8" name="Predictions" />
              <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#0066CC" name="Accuracy %" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;