import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AgentCollaboration, MeetingMinutes } from '@/types';
import { Download, FileText, Users, Target, Briefcase, User } from 'lucide-react';

interface MeetingMinutesModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  collaboration: AgentCollaboration | null;
}

const MeetingMinutesModal: React.FC<MeetingMinutesModalProps> = ({
  open,
  onOpenChange,
  collaboration
}) => {
  const [activeVersion, setActiveVersion] = useState('executive');
  const [minutes, setMinutes] = useState<Record<string, MeetingMinutes>>({});

  // 模拟生成不同版本的会议纪要
  useEffect(() => {
    if (collaboration && collaboration.currentPhase === 'completed') {
      const mockMinutes = generateMockMinutes(collaboration);
      setMinutes(mockMinutes);
    }
  }, [collaboration]);

  const generateMockMinutes = (collab: AgentCollaboration): Record<string, MeetingMinutes> => {
    const baseContent = collab.results || {};
    
    return {
      executive: {
        id: 'exec-1',
        meetingId: collab.sessionId,
        version: 'executive',
        content: `# 高管版会议纪要

## 执行摘要
本次会议围绕产品功能优化展开，确定了以用户体验为核心的改进策略，并明确了50万元的预算框架。

## 关键决策
1. **战略方向**: 确立用户体验优先的产品优化策略
2. **资源配置**: 批准50万元预算用于功能优化项目
3. **时间规划**: 项目预计在下个季度完成主要功能改进

## 商业影响
- **市场机会**: 用户体验提升将增强产品竞争力
- **风险控制**: 严格的预算管控确保项目可控性
- **投资回报**: 预期用户满意度提升15-20%

## 下一步行动
- 李四负责用户体验调研，下周五前完成
- 王五制定详细预算方案，本周末提交
- 张三协调技术资源，下周三前确认方案可行性`,
        keyPoints: ['用户体验优化', '50万预算控制', '季度目标达成'],
        actionItems: [
          {
            id: '1',
            task: '用户体验调研报告',
            assignee: '李四',
            deadline: '下周五',
            priority: 'high',
            status: 'pending'
          }
        ],
        decisions: ['确定用户体验优先策略', '批准50万预算'],
        createdAt: new Date().toISOString(),
        generatedBy: collab.agents.filter(a => ['analyst-agent', 'editor-agent'].includes(a.id))
      },
      
      technical: {
        id: 'tech-1',
        meetingId: collab.sessionId,
        version: 'technical',
        content: `# 技术版会议纪要

## 技术需求分析
基于用户体验优化目标，技术团队需要重点关注以下领域：

### 前端优化
- 用户界面响应速度提升
- 交互体验流畅性改进
- 移动端适配优化

### 后端架构
- API性能优化
- 数据处理效率提升
- 系统稳定性增强

### 技术预算分配
- 前端开发: 20万元
- 后端优化: 15万元
- 测试验证: 10万元
- 基础设施: 5万元

## 实施方案
1. **第一阶段**: 用户调研和需求分析 (2周)
2. **第二阶段**: 技术方案设计 (1周)  
3. **第三阶段**: 开发实施 (4周)
4. **第四阶段**: 测试和部署 (1周)

## 技术风险评估
- 预算超支风险: 中等
- 技术难度风险: 低
- 时间延期风险: 低

## 所需资源
- 前端工程师: 2人
- 后端工程师: 2人
- UI/UX设计师: 1人
- 测试工程师: 1人`,
        keyPoints: ['前端用户体验', '后端性能优化', '技术方案设计'],
        actionItems: [
          {
            id: '2',
            task: '技术方案设计文档',
            assignee: '张三',
            deadline: '下周三',
            priority: 'high',
            status: 'pending'
          }
        ],
        decisions: ['技术架构确认', '开发资源分配'],
        createdAt: new Date().toISOString(),
        generatedBy: collab.agents.filter(a => ['recorder-agent', 'analyst-agent'].includes(a.id))
      },

      management: {
        id: 'mgmt-1',
        meetingId: collab.sessionId,
        version: 'management',
        content: `# 管理版会议纪要

## 项目管理概要
本次会议明确了产品优化项目的管理框架和执行计划。

## 团队分工
### 李四 - 用户体验负责人
- 负责用户调研和体验分析
- 制定用户体验改进标准
- 协调设计团队资源

### 王五 - 财务管理
- 预算制定和控制
- 成本分析和报告
- 财务风险评估

### 张三 - 技术总监
- 技术方案设计
- 开发团队管理
- 技术风险控制

## 项目里程碑
1. **里程碑1**: 需求确认 (第2周)
2. **里程碑2**: 方案评审 (第3周)
3. **里程碑3**: 开发完成 (第7周)
4. **里程碑4**: 上线部署 (第8周)

## 质量控制
- 每周进度汇报
- 关键节点评审
- 用户测试验证

## 沟通机制
- 日常: 站立会议
- 周度: 进度回顾
- 月度: 高层汇报

## 成功指标
- 用户满意度提升 ≥ 15%
- 系统响应时间改善 ≥ 30%
- 项目按时完成率 = 100%`,
        keyPoints: ['团队协作', '项目管理', '质量控制'],
        actionItems: [
          {
            id: '3',
            task: '项目管理计划',
            assignee: '项目管理办',
            deadline: '下周一',
            priority: 'medium',
            status: 'pending'
          }
        ],
        decisions: ['项目管理框架', '团队分工确认'],
        createdAt: new Date().toISOString(),
        generatedBy: collab.agents.filter(a => ['secretary-agent', 'qa-agent'].includes(a.id))
      },

      client: {
        id: 'client-1',
        meetingId: collab.sessionId,
        version: 'client',
        content: `# 客户版会议纪要

## 项目背景
为了提升用户体验和产品价值，我们启动了全面的产品功能优化项目。

## 客户价值
### 直接收益
- 产品使用体验显著改善
- 功能响应速度大幅提升
- 界面更加美观易用

### 长期价值
- 降低学习成本
- 提高工作效率
- 增强产品粘性

## 投资说明
项目总投资50万元，主要用于：
- 用户体验研究和设计
- 技术架构优化升级
- 质量测试和验证

## 时间计划
- **启动阶段**: 即日起2周内
- **开发阶段**: 接下来6周
- **验收阶段**: 最后1周
- **预计完成**: 8周后

## 预期成果
1. **性能提升**: 系统响应速度提升30%以上
2. **体验改善**: 用户界面和交互全面优化
3. **稳定性**: 系统稳定性和可靠性显著增强

## 质量保证
- 严格的测试流程
- 用户验收确认
- 持续的技术支持

## 后续服务
- 3个月免费技术支持
- 用户培训和指导
- 定期系统维护`,
        keyPoints: ['客户价值', '投资回报', '服务保障'],
        actionItems: [
          {
            id: '4',
            task: '客户沟通方案',
            assignee: '客户成功团队',
            deadline: '明天',
            priority: 'high',
            status: 'pending'
          }
        ],
        decisions: ['客户价值确认', '服务标准制定'],
        createdAt: new Date().toISOString(),
        generatedBy: collab.agents.filter(a => ['editor-agent', 'secretary-agent'].includes(a.id))
      }
    };
  };

  const getVersionIcon = (version: string) => {
    switch (version) {
      case 'executive':
        return <Briefcase className="w-4 h-4" />;
      case 'technical':
        return <Target className="w-4 h-4" />;
      case 'management':
        return <Users className="w-4 h-4" />;
      case 'client':
        return <User className="w-4 h-4" />;
      default:
        return <FileText className="w-4 h-4" />;
    }
  };

  const getVersionName = (version: string) => {
    switch (version) {
      case 'executive':
        return '高管版';
      case 'technical':
        return '技术版';
      case 'management':
        return '管理版';
      case 'client':
        return '客户版';
      default:
        return '通用版';
    }
  };

  const downloadMinutes = (version: string) => {
    const minute = minutes[version];
    if (!minute) return;

    const content = minute.content;
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `会议纪要-${getVersionName(version)}-${new Date().toLocaleDateString()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  if (!collaboration || collaboration.currentPhase !== 'completed') {
    return null;
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            多版本会议纪要
          </DialogTitle>
        </DialogHeader>

        <Tabs value={activeVersion} onValueChange={setActiveVersion} className="w-full">
          <div className="flex justify-between items-center mb-4">
            <TabsList className="grid w-full max-w-md grid-cols-4">
              <TabsTrigger value="executive" className="flex items-center gap-1">
                {getVersionIcon('executive')}
                <span className="hidden sm:inline">高管版</span>
              </TabsTrigger>
              <TabsTrigger value="technical" className="flex items-center gap-1">
                {getVersionIcon('technical')}
                <span className="hidden sm:inline">技术版</span>
              </TabsTrigger>
              <TabsTrigger value="management" className="flex items-center gap-1">
                {getVersionIcon('management')}
                <span className="hidden sm:inline">管理版</span>
              </TabsTrigger>
              <TabsTrigger value="client" className="flex items-center gap-1">
                {getVersionIcon('client')}
                <span className="hidden sm:inline">客户版</span>
              </TabsTrigger>
            </TabsList>

            <Button
              onClick={() => downloadMinutes(activeVersion)}
              className="flex items-center gap-2"
              disabled={!minutes[activeVersion]}
            >
              <Download className="w-4 h-4" />
              下载纪要
            </Button>
          </div>

          {Object.entries(minutes).map(([version, minute]) => (
            <TabsContent key={version} value={version} className="space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                {/* 主要内容 */}
                <div className="lg:col-span-2">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        {getVersionIcon(version)}
                        {getVersionName(version)}纪要内容
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="prose prose-sm max-w-none">
                        <pre className="whitespace-pre-wrap text-sm leading-relaxed">
                          {minute.content}
                        </pre>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* 侧边栏信息 */}
                <div className="space-y-4">
                  {/* 关键要点 */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">关键要点</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2">
                      {minute.keyPoints.map((point, index) => (
                        <Badge key={index} variant="outline" className="block text-center">
                          {point}
                        </Badge>
                      ))}
                    </CardContent>
                  </Card>

                  {/* 待办事项 */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">待办事项</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {minute.actionItems.map((item) => (
                        <div key={item.id} className="p-3 bg-gray-50 rounded-lg space-y-2">
                          <p className="font-medium text-sm">{item.task}</p>
                          <div className="flex justify-between text-xs text-gray-600">
                            <span>负责人: {item.assignee}</span>
                            <span>{item.deadline}</span>
                          </div>
                          <Badge 
                            variant={item.priority === 'high' ? 'destructive' : 'secondary'}
                            className="text-xs"
                          >
                            {item.priority === 'high' ? '高优先级' : 
                             item.priority === 'medium' ? '中优先级' : '低优先级'}
                          </Badge>
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                  {/* 生成信息 */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="text-sm">生成信息</CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-2 text-xs text-gray-600">
                      <p>生成时间: {new Date(minute.createdAt).toLocaleString()}</p>
                      <p>参与智能体: {minute.generatedBy.length}个</p>
                      <div className="flex flex-wrap gap-1">
                        {minute.generatedBy.map((agent) => (
                          <Badge key={agent.id} variant="outline" className="text-xs">
                            {agent.name}
                          </Badge>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </DialogContent>
    </Dialog>
  );
};

export default MeetingMinutesModal;