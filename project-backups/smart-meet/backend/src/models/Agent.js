// 智能体模型
export class Agent {
  constructor(data) {
    this.id = data.id;
    this.name = data.name;
    this.role = data.role;
    this.status = data.status || 'idle'; // idle, working, completed, error
    this.progress = data.progress || 0;
    this.description = data.description;
    this.capabilities = data.capabilities || [];
    this.modelType = data.modelType; // qianwen, deepseek
    this.config = data.config || {};
    this.createdAt = data.createdAt || new Date().toISOString();
    this.updatedAt = data.updatedAt || new Date().toISOString();
  }

  updateStatus(status, progress = null) {
    this.status = status;
    if (progress !== null) {
      this.progress = progress;
    }
    this.updatedAt = new Date().toISOString();
  }

  toJSON() {
    return {
      id: this.id,
      name: this.name,
      role: this.role,
      status: this.status,
      progress: this.progress,
      description: this.description,
      capabilities: this.capabilities,
      modelType: this.modelType,
      config: this.config,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt
    };
  }
}

export default Agent;