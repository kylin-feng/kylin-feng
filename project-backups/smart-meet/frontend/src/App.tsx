import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import MinimalLandingPage from './pages/MinimalLandingPage';
import MinimalDashboard from './pages/MinimalDashboard';
import LandingPage from './pages/LandingPage';
import EnhancedDashboard from './pages/EnhancedDashboard';
import './index.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<MinimalLandingPage />} />
          <Route path="/dashboard" element={<MinimalDashboard />} />
          {/* 原版页面保留作为备选 */}
          <Route path="/original" element={<LandingPage />} />
          <Route path="/enhanced" element={<EnhancedDashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;