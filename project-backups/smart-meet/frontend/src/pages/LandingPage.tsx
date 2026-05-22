import React from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useNavigate } from 'react-router-dom';
import Navigation from '@/components/layout/Navigation';
import {
  Brain,
  Mic,
  Users,
  FileText,
  Zap,
  ChevronRight,
  Play,
  CheckCircle,
  Clock,
  Target,
  Sparkles,
  ArrowRight
} from 'lucide-react';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();

  const features = [
    {
      icon: <Mic className="w-8 h-8 text-blue-400" />,
      title: "拯救记录恐惧症",
      description: "AI全程自动记录，再也不用边听边记到手抽筋，专心开会就够了"
    },
    {
      icon: <Users className="w-8 h-8 text-green-400" />,
      title: "治愈懒癌晚期",
      description: "5个AI打工仔替你干活，比最勤快的实习生还靠谱，永远不摸鱼"
    },
    {
      icon: <FileText className="w-8 h-8 text-purple-400" />,
      title: "秒杀拖延症",
      description: "会议结束纪要就出来，再也不会'明天整理'拖到下周，强迫症都治好了"
    },
    {
      icon: <Zap className="w-8 h-8 text-yellow-400" />,
      title: "终结扯皮大战",
      description: "谁说什么一清二楚，任务分配明明白白，想甩锅都没机会"
    }
  ];

  const stats = [
    { number: "90%", label: "拖延症治愈率", icon: <Clock className="w-6 h-6" /> },
    { number: "0次", label: "会后催纪要", icon: <Target className="w-6 h-6" /> },
    { number: "5个", label: "AI苦力", icon: <Brain className="w-6 h-6" /> },
    { number: "100%", label: "懒人友好", icon: <Sparkles className="w-6 h-6" /> }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      <Navigation />
      {/* Hero Section */}
      <div className="relative overflow-hidden">
        {/* Background Animation */}
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500 rounded-full blur-3xl animate-pulse"></div>
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-blue-500 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }}></div>
        </div>
        
        <div className="relative container mx-auto px-6 py-20 pt-32">
          <div className="text-center space-y-8 max-w-4xl mx-auto">
            {/* Logo & Brand */}
            <div className="flex items-center justify-center gap-4 mb-8">
              <div className="p-4 bg-gradient-to-r from-purple-500 to-pink-500 rounded-2xl">
                <Brain className="w-12 h-12 text-white" />
              </div>
              <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-blue-400 bg-clip-text text-transparent">
                SmartMeet AI
              </h1>
            </div>

            {/* Core Pain Point */}
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-6 mb-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Target className="w-6 h-6 text-red-400" />
                <span className="text-red-400 font-semibold text-lg">职场效率杀手</span>
              </div>
              <div className="grid md:grid-cols-2 gap-4 text-white">
                <div className="space-y-2">
                  <p className="text-xl font-bold text-red-300">😴 拖延症爆发</p>
                  <p className="text-base">会后拖延整理，纪要一拖再拖，重要事项被遗忘</p>
                </div>
                <div className="space-y-2">
                  <p className="text-xl font-bold text-red-300">🐌 懒惰病严重</p>
                  <p className="text-base">手工记录太累，整理格式嫌麻烦，能拖就拖</p>
                </div>
                <div className="space-y-2">
                  <p className="text-xl font-bold text-red-300">⏰ 时间黑洞</p>
                  <p className="text-base">2小时会议+3小时整理，效率极低还容易出错</p>
                </div>
                <div className="space-y-2">
                  <p className="text-xl font-bold text-red-300">🤯 协作混乱</p>
                  <p className="text-base">谁说了什么记不清，任务分配不明确，推诿扯皮</p>
                </div>
              </div>
            </div>

            {/* Product Introduction */}
            <div className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/30 rounded-xl p-8">
              <div className="flex items-center justify-center gap-3 mb-4">
                <Sparkles className="w-6 h-6 text-purple-400" />
                <span className="text-purple-400 font-semibold text-lg">懒人福音 - 终结职场低效</span>
              </div>
              <p className="text-3xl font-bold text-white leading-relaxed mb-4">
                AI帮你克服拖延症，彻底告别开会低效率
              </p>
              <p className="text-xl text-gray-300 leading-relaxed mb-6">
                多智能体24小时不偷懒，自动搞定一切会议记录工作，让拖延症患者也能秒变效率达人
              </p>
              <div className="grid md:grid-cols-3 gap-4 text-center">
                <div className="bg-green-500/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-green-300">0秒</p>
                  <p className="text-sm text-gray-300">人工整理时间</p>
                </div>
                <div className="bg-blue-500/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-blue-300">5个</p>
                  <p className="text-sm text-gray-300">AI打工仔</p>
                </div>
                <div className="bg-purple-500/20 rounded-lg p-4">
                  <p className="text-2xl font-bold text-purple-300">100%</p>
                  <p className="text-sm text-gray-300">治愈拖延症</p>
                </div>
              </div>
            </div>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row gap-4 justify-center items-center pt-8">
              <Button 
                onClick={() => navigate('/dashboard')}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-2xl group"
                size="lg"
              >
                <Play className="w-5 h-5 mr-2 group-hover:scale-110 transition-transform" />
                拯救我的拖延症
                <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </Button>
              
              <Button 
                variant="outline"
                className="border-purple-500 text-purple-300 hover:bg-purple-500/10 px-8 py-4 text-lg font-semibold rounded-xl transition-all duration-300"
                size="lg"
              >
                了解更多
                <ChevronRight className="w-5 h-5 ml-2" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Section */}
      <div className="py-16 bg-black/20">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <Card key={index} className="glass-effect text-center border-none">
                <CardContent className="p-6 space-y-3">
                  <div className="flex justify-center text-purple-400">
                    {stat.icon}
                  </div>
                  <div className="text-3xl font-bold text-white">{stat.number}</div>
                  <div className="text-gray-400">{stat.label}</div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <Badge className="bg-purple-500/20 text-purple-300 border-purple-500/50 mb-4">
              职场救星功能
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">
              专治各种职场"疑难杂症"
            </h2>
            <p className="text-xl text-gray-300 max-w-2xl mx-auto">
              针对拖延症、懒惰病、效率低下等职场顽疾，AI开出特效药方
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-2 gap-8">
            {features.map((feature, index) => (
              <Card key={index} className="glass-effect border-none group hover:scale-105 transition-all duration-300">
                <CardContent className="p-8">
                  <div className="flex items-start space-x-4">
                    <div className="p-3 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-xl group-hover:scale-110 transition-transform">
                      {feature.icon}
                    </div>
                    <div className="flex-1">
                      <h3 className="text-xl font-semibold text-white mb-3">
                        {feature.title}
                      </h3>
                      <p className="text-gray-300 leading-relaxed">
                        {feature.description}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="py-20 bg-black/20">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16">
            <Badge className="bg-blue-500/20 text-blue-300 border-blue-500/50 mb-4">
              工作流程
            </Badge>
            <h2 className="text-4xl font-bold text-white mb-4">
              三步开启智能会议
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "01",
                title: "躺平开会",
                description: "啥也不用记，专心听就行，AI全程盯着",
                icon: <Play className="w-8 h-8" />
              },
              {
                step: "02", 
                title: "AI苦力干活",
                description: "5个AI拼命记录分析，比你勤快1000倍",
                icon: <Mic className="w-8 h-8" />
              },
              {
                step: "03",
                title: "秒出纪要",
                description: "会议结束，纪要就在手，拖延症没机会发作",
                icon: <FileText className="w-8 h-8" />
              }
            ].map((item, index) => (
              <div key={index} className="text-center group">
                <div className="relative mb-6">
                  <div className="w-20 h-20 bg-gradient-to-r from-purple-500 to-blue-500 rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform text-white">
                    {item.icon}
                  </div>
                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-yellow-500 text-black rounded-full flex items-center justify-center text-sm font-bold">
                    {item.step}
                  </div>
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">{item.title}</h3>
                <p className="text-gray-300">{item.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Final CTA Section */}
      <div className="py-20">
        <div className="container mx-auto px-6 text-center">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-4xl font-bold text-white mb-6">
              拖延症患者的福音来了！
            </h2>
            <p className="text-xl text-gray-300 mb-8">
              不想记录？懒得整理？那就让AI替你搞定一切，从此告别低效开会
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Button 
                onClick={() => navigate('/dashboard')}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-10 py-4 text-lg font-semibold rounded-xl transition-all duration-300 transform hover:scale-105 shadow-2xl"
                size="lg"
              >
                <Zap className="w-5 h-5 mr-2" />
                立即治愈拖延症
              </Button>
            </div>

            <div className="flex items-center justify-center gap-6 mt-8 text-gray-400">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span>懒人专用</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span>0秒上手</span>
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <span>拖延症克星</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="border-t border-gray-800 py-8">
        <div className="container mx-auto px-6 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            <Brain className="w-6 h-6 text-purple-400" />
            <span className="text-white font-semibold">SmartMeet AI</span>
          </div>
          <p className="text-gray-400">
            © 2024 SmartMeet AI. 让每一次会议都产生价值.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;