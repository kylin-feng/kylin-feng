import express from 'express';
import AgentController from '../controllers/AgentController.js';

const router = express.Router();

// 智能体路由
router.get('/', AgentController.getAgents);
router.get('/:agentId', AgentController.getAgent);
router.put('/:agentId/status', AgentController.updateAgentStatus);

// 协作路由
router.post('/collaborate', AgentController.startCollaboration);
router.get('/collaborate/:sessionId', AgentController.getCollaborationStatus);
router.post('/collaborate/:sessionId/stop', AgentController.stopCollaboration);

export default router;