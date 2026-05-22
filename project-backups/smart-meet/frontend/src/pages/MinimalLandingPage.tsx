import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';
import {
  Brain,
  Mic,
  Users,
  FileText,
  Zap,
  ArrowRight,
  CheckCircle
} from 'lucide-react';

const MinimalLandingPage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Mic className="w-6 h-6" />,
      title: "自动记录",
      description: "AI全程记录，无需手工笔记"
    },
    {
      icon: <Users className="w-6 h-6" />,
      title: "智能协作",
      description: "多智能体分工合作，高效处理"
    },
    {
      icon: <FileText className="w-6 h-6" />,
      title: "即时纪要",
      description: "会议结束，纪要即刻生成"
    },
    {
      icon: <Zap className="w-6 h-6" />,
      title: "零延迟",
      description: "告别拖延，提升会议效率"
    }
  ];

  const benefits = [
    "90% 拖延症治愈率",
    "0次 会后催纪要", 
    "5个 AI助手同时工作",
    "1秒 生成多版本纪要"
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-slate-100">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-slate-900 rounded-lg flex items-center justify-center">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <span className="text-lg font-medium text-slate-900">SmartMeet AI</span>
            </div>
            <Button variant="outline" size="sm" className="rounded-full">
              登录
            </Button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="px-6 py-24">
        <div className="max-w-4xl mx-auto text-center">
          <Badge className="mb-8 bg-slate-100 text-slate-700 border-0 rounded-full px-4 py-2">
            专治职场拖延症
          </Badge>
          
          <h1 className="text-5xl font-light text-slate-900 mb-6 tracking-tight">
            让会议变得
            <span className="font-medium"> 简单高效</span>
          </h1>
          
          <p className="text-lg text-slate-600 mb-12 max-w-2xl mx-auto font-light leading-relaxed">
            AI驱动的智能会议助手，自动记录、实时分析、即时生成纪要
          </p>

          <div className="flex items-center justify-center gap-4 mb-16">
            <Button 
              size="lg" 
              className="rounded-full px-8 bg-slate-900 hover:bg-slate-800"
              onClick={() => navigate('/dashboard')}
            >
              开始体验
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
            <Button 
              variant="outline" 
              size="lg" 
              className="rounded-full px-8"
            >
              观看演示
            </Button>
          </div>

          {/* Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 mb-20">
            {benefits.map((benefit, index) => (
              <div key={index} className="text-center">
                <div className="text-2xl font-light text-slate-900 mb-1">
                  {benefit.split(' ')[0]}
                </div>
                <div className="text-sm text-slate-500 font-light">
                  {benefit.split(' ').slice(1).join(' ')}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="px-6 py-20 bg-slate-50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-light text-slate-900 mb-4">
              核心功能
            </h2>
            <p className="text-slate-600 font-light">
              简化会议流程，提升工作效率
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {features.map((feature, index) => (
              <Card key={index} className="border-0 shadow-none bg-white hover:shadow-sm transition-shadow">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 bg-slate-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    {feature.icon}
                  </div>
                  <h3 className="font-medium text-slate-900 mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-slate-600 font-light leading-relaxed">
                    {feature.description}
                  </p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="px-6 py-20">
        <div className="max-w-4xl mx-auto text-center">
          <Card className="border-0 shadow-none bg-slate-900 text-white">
            <CardContent className="p-12">
              <h2 className="text-3xl font-light mb-4">
                准备好提升会议效率了吗？
              </h2>
              <p className="text-slate-300 mb-8 font-light">
                加入数千家企业，体验AI驱动的智能会议管理
              </p>
              <Button 
                size="lg" 
                className="rounded-full px-8 bg-white text-slate-900 hover:bg-slate-100"
                onClick={() => navigate('/dashboard')}
              >
                立即开始
                <ArrowRight className="w-4 h-4 ml-2" />
              </Button>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-slate-100 px-6 py-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-6 h-6 bg-slate-900 rounded flex items-center justify-center">
                <Brain className="w-4 h-4 text-white" />
              </div>
              <span className="text-sm text-slate-600">SmartMeet AI</span>
            </div>
            <div className="text-sm text-slate-500">
              © 2024 SmartMeet AI. 让会议更简单.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default MinimalLandingPage;