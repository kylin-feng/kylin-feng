const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 启动 SmartMeet AI 服务器...\n');

// 杀死现有进程
try {
  require('child_process').execSync('pkill -f "node.*index.js"', { stdio: 'ignore' });
  require('child_process').execSync('pkill -f "vite"', { stdio: 'ignore' });
  require('child_process').execSync('pkill -f "nodemon"', { stdio: 'ignore' });
} catch (e) {
  // 进程可能不存在
}

console.log('📊 启动后端服务器 (端口 5001)...');

// 启动后端
const backend = spawn('node', ['src/index.js'], {
  cwd: path.join(__dirname, 'backend'),
  stdio: 'pipe'
});

backend.stdout.on('data', (data) => {
  console.log(`[后端] ${data}`);
});

backend.stderr.on('data', (data) => {
  console.error(`[后端错误] ${data}`);
});

// 等待后端启动
setTimeout(() => {
  console.log('\n🎨 启动前端服务器 (端口 3000)...');
  
  // 启动前端
  const frontend = spawn('npx', ['vite'], {
    cwd: path.join(__dirname, 'frontend'),
    stdio: 'pipe'
  });

  frontend.stdout.on('data', (data) => {
    console.log(`[前端] ${data}`);
  });

  frontend.stderr.on('data', (data) => {
    console.error(`[前端错误] ${data}`);
  });

  frontend.on('error', (err) => {
    console.error('前端启动失败:', err);
  });

}, 3000);

backend.on('error', (err) => {
  console.error('后端启动失败:', err);
});

// 处理清理
process.on('SIGINT', () => {
  console.log('\n⚠️  正在关闭服务...');
  try {
    require('child_process').execSync('pkill -f "node.*index.js"', { stdio: 'ignore' });
    require('child_process').execSync('pkill -f "vite"', { stdio: 'ignore' });
  } catch (e) {}
  process.exit(0);
});

console.log('\n✅ 正在启动服务...');
console.log('前端应用: http://localhost:3000');
console.log('后端API: http://localhost:5001/api/health');
console.log('\n按 Ctrl+C 停止所有服务\n');