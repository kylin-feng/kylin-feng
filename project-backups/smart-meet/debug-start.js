#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

console.log('🚀 Starting SmartMeet AI Debug Mode...\n');

// Kill existing processes
const killProcesses = () => {
  try {
    require('child_process').execSync('pkill -f "vite"', { stdio: 'ignore' });
    require('child_process').execSync('pkill -f "nodemon"', { stdio: 'ignore' });
  } catch (e) {
    // Processes might not exist
  }
};

killProcesses();

// Start backend
console.log('📊 Starting Backend Server...');
const backend = spawn('npm', ['run', 'dev'], {
  cwd: path.join(__dirname, 'backend'),
  stdio: 'inherit',
  shell: true
});

// Wait a bit for backend to start
setTimeout(() => {
  console.log('\n🎨 Starting Frontend Server...');
  const frontend = spawn('npm', ['run', 'dev'], {
    cwd: path.join(__dirname, 'frontend'),
    stdio: 'inherit',
    shell: true
  });

  frontend.on('error', (err) => {
    console.error('Frontend error:', err);
  });
}, 3000);

backend.on('error', (err) => {
  console.error('Backend error:', err);
});

// Handle cleanup
process.on('SIGINT', () => {
  console.log('\n⚠️  Shutting down services...');
  killProcesses();
  process.exit(0);
});

console.log('\n✅ Services are starting...');
console.log('Frontend: http://localhost:3000');
console.log('Backend API: http://localhost:5001');
console.log('\nPress Ctrl+C to stop all services\n');