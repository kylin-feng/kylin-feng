import React from 'react';
import { Button } from '@/components/ui/button';
import { Brain, Menu } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface MinimalNavigationProps {
  variant?: 'light' | 'dark';
}

const MinimalNavigation: React.FC<MinimalNavigationProps> = ({ 
  variant = 'light' 
}) => {
  const navigate = useNavigate();

  const isDark = variant === 'dark';

  return (
    <nav className={`border-b ${isDark ? 'border-slate-800 bg-slate-900' : 'border-slate-100 bg-white'}`}>
      <div className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between">
          {/* Logo */}
          <div 
            className="flex items-center gap-3 cursor-pointer" 
            onClick={() => navigate('/')}
          >
            <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
              isDark ? 'bg-white' : 'bg-slate-900'
            }`}>
              <Brain className={`w-5 h-5 ${isDark ? 'text-slate-900' : 'text-white'}`} />
            </div>
            <span className={`text-lg font-medium ${
              isDark ? 'text-white' : 'text-slate-900'
            }`}>
              SmartMeet AI
            </span>
          </div>

          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button 
              variant="ghost" 
              size="sm" 
              className={`hidden md:flex rounded-full ${
                isDark ? 'text-slate-300 hover:text-white' : 'text-slate-600 hover:text-slate-900'
              }`}
              onClick={() => navigate('/dashboard')}
            >
              控制台
            </Button>
            <Button 
              variant={isDark ? "outline" : "outline"} 
              size="sm" 
              className={`rounded-full ${
                isDark 
                  ? 'border-slate-700 text-slate-300 hover:bg-slate-800' 
                  : 'border-slate-200 text-slate-700 hover:bg-slate-50'
              }`}
            >
              登录
            </Button>
            <Button 
              variant="ghost" 
              size="sm" 
              className={`md:hidden rounded-full ${
                isDark ? 'text-slate-300' : 'text-slate-600'
              }`}
            >
              <Menu className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default MinimalNavigation;