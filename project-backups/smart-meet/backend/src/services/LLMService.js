import axios from 'axios';
import config from '../config/index.js';

// 大语言模型服务
export class LLMService {
  constructor() {
    this.qianwenClient = axios.create({
      baseURL: config.qianwen.apiUrl,
      headers: {
        'Authorization': `Bearer ${config.qianwen.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });

    this.deepseekClient = axios.create({
      baseURL: config.deepseek.apiUrl,
      headers: {
        'Authorization': `Bearer ${config.deepseek.apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 30000
    });
  }

  // 调用通义千问API
  async callQianwen(prompt, options = {}) {
    try {
      // 如果没有配置API密钥，返回模拟结果
      if (!config.qianwen.apiKey || config.qianwen.apiKey === 'your_qianwen_api_key_here') {
        return this.getMockQianwenResponse(prompt);
      }

      const requestData = {
        model: options.model || 'qwen-turbo',
        input: {
          messages: [
            {
              role: 'system',
              content: '你是SmartMeet AI的专业智能体，专门负责会议内容处理。请按照要求认真分析和处理会议内容。'
            },
            {
              role: 'user', 
              content: prompt
            }
          ]
        },
        parameters: {
          temperature: options.temperature || 0.3,
          max_tokens: options.maxTokens || 2000,
          top_p: options.topP || 0.8
        }
      };

      const response = await this.qianwenClient.post('', requestData);
      
      if (response.data && response.data.output && response.data.output.text) {
        return {
          success: true,
          content: response.data.output.text,
          model: 'qwen-turbo',
          usage: response.data.usage || {}
        };
      } else {
        throw new Error('Invalid response format from Qianwen API');
      }

    } catch (error) {
      console.error('通义千问API调用失败:', error.message);
      
      // 返回模拟结果作为降级方案
      return this.getMockQianwenResponse(prompt);
    }
  }

  // 调用DeepSeek API
  async callDeepSeek(prompt, options = {}) {
    try {
      // 如果没有配置API密钥，返回模拟结果
      if (!config.deepseek.apiKey || config.deepseek.apiKey === 'your_deepseek_api_key_here') {
        return this.getMockDeepSeekResponse(prompt);
      }

      const requestData = {
        model: options.model || 'deepseek-chat',
        messages: [
          {
            role: 'system',
            content: '你是SmartMeet AI的质量检查专家，专门负责会议内容的逻辑验证和准确性检查。'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: options.temperature || 0.1,
        max_tokens: options.maxTokens || 2000,
        top_p: options.topP || 0.95
      };

      const response = await this.deepseekClient.post('', requestData);

      if (response.data && response.data.choices && response.data.choices[0]) {
        return {
          success: true,
          content: response.data.choices[0].message.content,
          model: 'deepseek-chat',
          usage: response.data.usage || {}
        };
      } else {
        throw new Error('Invalid response format from DeepSeek API');
      }

    } catch (error) {
      console.error('DeepSeek API调用失败:', error.message);
      
      // 返回模拟结果作为降级方案
      return this.getMockDeepSeekResponse(prompt);
    }
  }

  // 通义千问模拟响应
  getMockQianwenResponse(prompt) {
    const responses = {
      recorder: `## 会议记录整理

**时间**: ${new Date().toLocaleString()}
**参与者**: 张三、李四、王五

### 会议内容
1. **张三** (10:00): 今天我们讨论的主要议题是产品功能优化，希望大家能够积极参与讨论。

2. **李四** (10:02): 我认为我们应该重点关注用户体验方面的改进，这是当前最需要优化的部分。

3. **王五** (10:05): 预算方面我们需要控制在50万以内，请大家在提出方案时考虑成本因素。

### 记录说明
- 会议录音质量良好，语音识别准确率约95%
- 已自动标注发言人和时间戳
- 修正了部分语音识别错误`,

      analyst: `## 会议内容分析报告

### 主要议题
1. **产品功能优化** - 会议核心主题
2. **用户体验改进** - 重点关注领域  
3. **预算控制** - 约束条件

### 关键决策
- 确定以用户体验为优化重点
- 设定预算上限为50万元
- 需要在提案中平衡功能与成本

### 重要观点
- **张三**: 强调讨论的重要性和参与度
- **李四**: 聚焦用户体验，认为这是优先级最高的改进方向
- **王五**: 财务约束明确，成本控制是关键因素

### 风险与机会
**机会**: 用户体验改进可能带来用户满意度提升
**风险**: 预算限制可能影响功能实现的完整性`,

      secretary: `## 待办事项清单

### 高优先级任务
1. **用户体验调研**
   - 负责人: 李四
   - 截止时间: 下周五
   - 内容: 收集用户反馈，分析体验痛点

2. **预算方案制定**
   - 负责人: 王五  
   - 截止时间: 本周末
   - 内容: 详细预算分解，成本控制方案

3. **功能优化方案**
   - 负责人: 张三
   - 截止时间: 下周三
   - 内容: 基于调研结果制定具体优化方案

### 下次会议安排
- 时间: 下周一 10:00
- 议题: 方案评审和最终决策
- 参与者: 全体成员

### 跟进事项
- 李四需要在周三前提供初步调研结果
- 王五需要确认预算审批流程
- 张三负责协调技术团队资源`,

      editor: `## 产品功能优化会议纪要

**会议时间**: ${new Date().toLocaleString()}
**会议地点**: 线上会议室
**参与人员**: 张三(项目负责人)、李四(用户体验专家)、王五(财务经理)

### 会议主题
本次会议主要围绕产品功能优化策略展开深入讨论，旨在确定优化方向和实施方案。

### 讨论要点

#### 1. 优化重点确认
李四同志强调，当前产品最需要改进的是用户体验方面，这应该成为我们的首要关注点。用户体验的提升将直接影响产品的市场竞争力和用户满意度。

#### 2. 预算约束明确
王五同志明确指出，本次优化项目的预算上限为人民币50万元。所有方案的制定都必须在此预算框架内进行，确保项目的经济可行性。

#### 3. 实施策略讨论
张三同志主持讨论，强调了全员参与的重要性，希望各部门能够通力合作，确保优化方案的顺利实施。

### 会议结论
会议达成一致意见，将以用户体验改进为核心，在50万元预算范围内制定具体的产品功能优化方案。`,

      qa: `## 质量检查报告

### 逻辑一致性检查
✅ **通过**: 会议讨论逻辑清晰，从问题识别到预算约束，形成了完整的决策链条

### 信息准确性验证
✅ **通过**: 关键信息明确
- 预算限制: 50万元 (具体数值)
- 优化重点: 用户体验 (明确方向)
- 责任分工: 各角色职责清晰

### 决策合理性分析
✅ **基本合理**: 
- 以用户体验为优先级符合产品发展逻辑
- 预算约束设定合理，避免无限制投入
- 参与者角色分工明确

### 潜在风险识别
⚠️ **需要注意**:
1. 50万预算是否充足需要进一步评估
2. 用户体验改进的具体指标和衡量标准未明确
3. 项目时间线较紧，需要确认资源可用性

### 补充建议
1. 建议设定明确的用户体验改进KPI指标
2. 需要制定详细的项目时间表和里程碑
3. 建议增加风险应对预案的讨论`
    };

    // 根据prompt内容返回对应的模拟结果
    if (prompt.includes('整理和格式化')) return { success: true, content: responses.recorder, model: 'qwen-mock' };
    if (prompt.includes('分析以下会议内容')) return { success: true, content: responses.analyst, model: 'qwen-mock' };
    if (prompt.includes('待办事项')) return { success: true, content: responses.secretary, model: 'qwen-mock' };
    if (prompt.includes('语言优化')) return { success: true, content: responses.editor, model: 'qwen-mock' };
    
    return { 
      success: true, 
      content: '模拟处理完成。在实际环境中，这里将返回通义千问的真实响应结果。', 
      model: 'qwen-mock' 
    };
  }

  // DeepSeek模拟响应
  getMockDeepSeekResponse(prompt) {
    if (prompt.includes('质量检查')) {
      return {
        success: true,
        content: `## 质量检查报告

### 逻辑一致性检查
✅ **通过**: 会议讨论逻辑清晰，从问题识别到预算约束，形成了完整的决策链条

### 信息准确性验证  
✅ **通过**: 关键信息明确
- 预算限制: 50万元 (具体数值)
- 优化重点: 用户体验 (明确方向)
- 责任分工: 各角色职责清晰

### 决策合理性分析
✅ **基本合理**: 
- 以用户体验为优先级符合产品发展逻辑
- 预算约束设定合理，避免无限制投入
- 参与者角色分工明确

### 潜在风险识别
⚠️ **需要注意**:
1. 50万预算是否充足需要进一步评估
2. 用户体验改进的具体指标和衡量标准未明确  
3. 项目时间线较紧，需要确认资源可用性

### 补充建议
1. 建议设定明确的用户体验改进KPI指标
2. 需要制定详细的项目时间表和里程碑
3. 建议增加风险应对预案的讨论`,
        model: 'deepseek-mock'
      };
    }

    return {
      success: true,
      content: '模拟质检完成。在实际环境中，这里将返回DeepSeek的真实分析结果。',
      model: 'deepseek-mock'
    };
  }

  // 获取模型状态
  async getModelStatus() {
    return {
      qianwen: {
        available: !!config.qianwen.apiKey && config.qianwen.apiKey !== 'your_qianwen_api_key_here',
        model: 'qwen-turbo'
      },
      deepseek: {
        available: !!config.deepseek.apiKey && config.deepseek.apiKey !== 'your_deepseek_api_key_here',
        model: 'deepseek-chat'
      }
    };
  }
}

export default LLMService;