# Project Backups

这里保存一些本机历史项目的干净备份版本，主要用于 GitHub 留档。

已排除内容：

- `node_modules/`、`venv/`、`dist/`、`build/` 等可再生成目录
- `.env`、`.streamlit/secrets.toml` 等本地密钥配置
- `.DS_Store`、`__pycache__/` 等系统或缓存文件

当前包含：

- `resume-optimizer/`：AI 简历优化助手，已移除硬编码 API Key
- `lifeLens-ai-assistant/`：Rokid AR 生活助手原型
- `xuanxuetool/`：玄学工具静态页面
- `lucky-color-generator/`：幸运色生成器
- `react-web/`：React + Vite 原型
- `tools-collection/`：本机工具脚本合集
- `zj-smart-meeting/`：智能会议工具原型
- `smart-meet/`：会议助手全栈原型，已排除本地 `.env`
- `zj-meeting/`：会议工具后端/前端原型，已替换硬编码 API Key
- `pet-companion-mactool/`：macOS 宠物陪伴工具原型
- `aiwrite-king/`：AI 写作 HarmonyOS 应用原型，已替换硬编码 API Key
- `rokid-skin-age-app/`：Rokid 皮肤年龄检测 Flutter 原型，已替换硬编码 API Key
- `haoduoai-qwen-image-test/`：通义万相图片生成测试页面，已替换硬编码 API Key
- `stanford-talent-town-pages/`：斯坦福人才小镇页面原型，已替换硬编码 API Key
- `work-web-edu/`：教育官网静态页面
- `work-web-lab/`：实验室官网静态页面
- `work-web-api/`：API 展示静态页面
- `work-web-guanwang/`：官网静态页面
- `q-ai-show/`：Q-AI 展示版前端项目
- `q-ai-edu/`：Q-AI 教育版前端项目
- `taro-app/`：Taro 应用原型
- `app-hm/`：鸿蒙应用原型
- `codex-red-quiz/`：Codex 生成的答题/测验项目
- `aiedu-app-center-prototype/`：AI 教育应用中心原型
- `work-test-app/`：Work-2025-10 里的 Flutter 测试应用，已排除构建产物
- `ai-between-us/`：Trae Solo Hackathon 关系/沟通 AI 原型，已替换 Coze PAT
- `between-us/`：Between-Us Web/AI 原型，已替换 Coze PAT
- `lucky-web/`：幸运色/运势 Web 原型
- `lucky-wuxing/`：五行/幸运色工具原型，已替换硬编码 API Key
- `p340-ai-answering/`：P340 机械臂 AI 答题项目，已替换配置里的 DeepSeek API Key

未纳入的内容：

- 部分 iCloud 占位文件读取超时的目录，避免备份成空壳
- 第三方大型源码、依赖缓存、构建产物和本地密钥配置
- 运行日志、虚拟环境、`data/input/`、`data/output/` 等本机运行数据
