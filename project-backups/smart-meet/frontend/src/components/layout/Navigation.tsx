import React from 'react';
import { Button } from '@/components/ui/button';
import { useNavigate, useLocation } from 'react-router-dom';
import { Brain, Home, BarChart3, ArrowLeft } from 'lucide-react';

const Navigation: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  
  const isOnDashboard = location.pathname === '/dashboard';

  return (
    <nav className="fixed top-4 left-4 right-4 z-50">
      <div className="glass-effect rounded-lg px-6 py-3">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div 
            className="flex items-center gap-3 cursor-pointer hover:scale-105 transition-transform"
            onClick={() => navigate('/')}
          >
            <div className="p-2 bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-white">SmartMeet AI</span>
          </div>

          {/* Navigation Buttons */}
          <div className="flex items-center gap-3">
            {isOnDashboard ? (
              <Button
                onClick={() => navigate('/')}
                variant="outline"
                className="border-gray-600 text-gray-300 hover:text-white hover:border-purple-500"
              >
                <ArrowLeft className="w-4 h-4 mr-2" />
                返回首页
              </Button>
            ) : (
              <>
                <Button
                  onClick={() => navigate('/dashboard')}
                  className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white"
                >
                  <BarChart3 className="w-4 h-4 mr-2" />
                  进入控制台
                </Button>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;