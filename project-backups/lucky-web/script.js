// 颜色定义映射
const COLOR_MAP = {
    1: { name: "绿色", color: "#4CAF50", description: "绿色代表生机与希望，今天适合开始新的计划" },
    2: { name: "青色", color: "#00BCD4", description: "青色象征清新与智慧，保持冷静思考会有好运" },
    3: { name: "红色", color: "#F44336", description: "红色充满热情与活力，大胆行动会带来机会" },
    4: { name: "紫色", color: "#9C27B0", description: "紫色神秘而高贵，直觉敏锐的一天" },
    5: { name: "黄色", color: "#FFC107", description: "黄色明亮温暖，乐观的心态会吸引好运" },
    6: { name: "棕色", color: "#795548", description: "棕色稳重踏实，脚踏实地会有收获" },
    7: { name: "白色", color: "#FFFFFF", description: "白色纯净简约，心境平和会带来好运" },
    8: { name: "金色", color: "#FFD700", description: "金色富贵吉祥，财运亨通的好日子" },
    9: { name: "黑色", color: "#212121", description: "黑色深邃神秘，内在力量觉醒的时刻" },
    10: { name: "蓝色", color: "#2196F3", description: "蓝色宁静深远，平静中蕴含无限可能" },
    11: { name: "灰色", color: "#9E9E9E", description: "灰色中庸平衡，保持低调会有意外收获" },
    12: { name: "银色", color: "#C0C0C0", description: "银色优雅高贵，贵人运势强劲" }
};

// 历史记录存储
let historyRecords = JSON.parse(localStorage.getItem('luckyColorHistory') || '[]');

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // 检查当前页面
    const currentPage = window.location.pathname.split('/').pop();
    
    if (currentPage === 'color-generator.html' || currentPage === '') {
        initializeGeneratorPage();
    }
    
    // 初始化导航栏活动状态
    updateNavigation();
});

// 初始化生成器页面
function initializeGeneratorPage() {
    const generateBtn = document.getElementById('generateBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultSection = document.getElementById('resultSection');
    const shareBtn = document.getElementById('shareBtn');
    const newGenerateBtn = document.getElementById('newGenerateBtn');
    
    if (generateBtn) {
        generateBtn.addEventListener('click', generateLuckyColor);
    }
    
    if (shareBtn) {
        shareBtn.addEventListener('click', shareResult);
    }
    
    if (newGenerateBtn) {
        newGenerateBtn.addEventListener('click', generateLuckyColor);
    }
    
    if (document.getElementById('hardwareBtn')) {
        document.getElementById('hardwareBtn').addEventListener('click', sendToHardware);
    }
    
    // 加载历史记录
    loadHistory();
}

// 生成幸运色
async function generateLuckyColor() {
    const generateBtn = document.getElementById('generateBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultSection = document.getElementById('resultSection');
    
    // 显示加载状态
    generateBtn.style.display = 'none';
    loadingSpinner.style.display = 'block';
    
    try {
        // 调用后端API
        const response = await fetch('http://10.10.100.19:5001/api/lucky-color');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 显示结果
        displayResult(data);
        
        // 添加到历史记录
        addToHistory(data);
        
        // 显示硬件发送状态
        if (data.hardware_sent) {
            showToast('🎯 幸运色已发送到硬件设备！');
        }
        
    } catch (error) {
        console.error('生成幸运色失败:', error);
        // 如果API失败，使用本地随机生成
        const fallbackData = generateFallbackColor();
        displayResult(fallbackData);
        addToHistory(fallbackData);
    } finally {
        // 隐藏加载状态
        loadingSpinner.style.display = 'none';
        generateBtn.style.display = 'inline-flex';
    }
}

// 显示结果
function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
    const resultDescription = document.getElementById('resultDescription');
    const colorDisplay = document.getElementById('colorDisplay');
    const colorMeaning = document.getElementById('colorMeaning');
    
    // 显示结果区域
    resultSection.style.display = 'block';
    
    // 设置描述
    resultDescription.textContent = data.description;
    
    // 清空颜色显示区域
    colorDisplay.innerHTML = '';
    
    // 显示颜色
    const colors = Array.isArray(data.color) ? data.color : [data.color];
    colors.forEach(colorId => {
        const colorInfo = COLOR_MAP[colorId];
        if (colorInfo) {
            const colorCircle = document.createElement('div');
            colorCircle.className = 'color-circle-result';
            colorCircle.style.background = `linear-gradient(45deg, ${colorInfo.color}, ${adjustBrightness(colorInfo.color, 20)})`;
            colorCircle.textContent = colorInfo.name;
            colorDisplay.appendChild(colorCircle);
        }
    });
    
    // 设置颜色寓意
    if (colors.length === 1) {
        const colorInfo = COLOR_MAP[colors[0]];
        colorMeaning.textContent = colorInfo ? colorInfo.description : '未知颜色寓意';
    } else {
        const colorNames = colors.map(id => COLOR_MAP[id]?.name || '未知').join('、');
        colorMeaning.textContent = `今日幸运色组合：${colorNames}，多重好运加持，事事顺心如意`;
    }
    
    // 滚动到结果区域
    resultSection.scrollIntoView({ behavior: 'smooth' });
}

// 调整颜色亮度
function adjustBrightness(hex, percent) {
    const num = parseInt(hex.replace("#", ""), 16);
    const amt = Math.round(2.55 * percent);
    const R = (num >> 16) + amt;
    const G = (num >> 8 & 0x00FF) + amt;
    const B = (num & 0x0000FF) + amt;
    return "#" + (0x1000000 + (R < 255 ? R < 1 ? 0 : R : 255) * 0x10000 +
        (G < 255 ? G < 1 ? 0 : G : 255) * 0x100 +
        (B < 255 ? B < 1 ? 0 : B : 255)).toString(16).slice(1);
}

// 生成备用颜色（当API不可用时）
function generateFallbackColor() {
    const colors = [];
    const numColors = Math.floor(Math.random() * 3) + 1;
    
    for (let i = 0; i < numColors; i++) {
        const colorId = Math.floor(Math.random() * 12) + 1;
        if (!colors.includes(colorId)) {
            colors.push(colorId);
        }
    }
    
    const colorNames = colors.map(id => COLOR_MAP[id]?.name || '未知').join('、');
    const description = colors.length === 1 
        ? COLOR_MAP[colors[0]]?.description || '未知颜色寓意'
        : `今日幸运色组合：${colorNames}，多重好运加持，事事顺心如意`;
    
    return {
        color: colors,
        description: description
    };
}

// 发送到硬件设备
async function sendToHardware() {
    const hardwareBtn = document.getElementById('hardwareBtn');
    if (!hardwareBtn) return;
    
    // 显示发送状态
    const originalText = hardwareBtn.innerHTML;
    hardwareBtn.innerHTML = '<span class="btn-icon">⏳</span>发送中...';
    hardwareBtn.disabled = true;
    
    try {
        const response = await fetch('http://10.10.100.19:5001/api/send-to-hardware', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        // 显示结果
        displayResult(data);
        
        // 显示发送状态
        if (data.hardware_sent) {
            showToast('🎯 幸运色已成功发送到硬件设备！');
        } else {
            showToast('⚠️ 发送到硬件设备失败，请检查连接');
        }
        
    } catch (error) {
        console.error('发送到硬件设备失败:', error);
        showToast('❌ 发送到硬件设备失败: ' + error.message);
    } finally {
        // 恢复按钮状态
        hardwareBtn.innerHTML = originalText;
        hardwareBtn.disabled = false;
    }
}

// 添加到历史记录
function addToHistory(data) {
    const record = {
        id: Date.now(),
        timestamp: new Date().toLocaleString('zh-CN'),
        colors: Array.isArray(data.color) ? data.color : [data.color],
        description: data.description
    };
    
    // 添加到历史记录开头
    historyRecords.unshift(record);
    
    // 只保留最近10条记录
    if (historyRecords.length > 10) {
        historyRecords = historyRecords.slice(0, 10);
    }
    
    // 保存到本地存储
    localStorage.setItem('luckyColorHistory', JSON.stringify(historyRecords));
    
    // 更新历史记录显示
    loadHistory();
}

// 加载历史记录
function loadHistory() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    if (historyRecords.length === 0) {
        historyList.innerHTML = '<p style="text-align: center; color: #999; font-style: italic;">暂无历史记录</p>';
        return;
    }
    
    historyRecords.forEach(record => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const colorsHtml = record.colors.map(colorId => {
            const colorInfo = COLOR_MAP[colorId];
            return `<div class="history-color" style="background: ${colorInfo?.color || '#ccc'}"></div>`;
        }).join('');
        
        historyItem.innerHTML = `
            ${colorsHtml}
            <div>
                <div style="font-weight: 600; color: #333;">${record.timestamp}</div>
                <div style="font-size: 0.8rem; color: #666;">${record.description}</div>
            </div>
        `;
        
        historyList.appendChild(historyItem);
    });
}

// 分享结果
function shareResult() {
    const resultDescription = document.getElementById('resultDescription');
    const shareText = `🎨 我的今日幸运色：${resultDescription.textContent}`;
    
    if (navigator.share) {
        navigator.share({
            title: '幸运色生成器',
            text: shareText,
            url: window.location.href
        }).catch(err => {
            console.log('分享失败:', err);
            copyToClipboard(shareText);
        });
    } else {
        copyToClipboard(shareText);
    }
}

// 复制到剪贴板
function copyToClipboard(text) {
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            showToast('结果已复制到剪贴板！');
        }).catch(err => {
            console.log('复制失败:', err);
            fallbackCopyToClipboard(text);
        });
    } else {
        fallbackCopyToClipboard(text);
    }
}

// 备用复制方法
function fallbackCopyToClipboard(text) {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
        document.execCommand('copy');
        showToast('结果已复制到剪贴板！');
    } catch (err) {
        console.log('复制失败:', err);
        showToast('复制失败，请手动复制');
    }
    document.body.removeChild(textArea);
}

// 显示提示消息
function showToast(message) {
    // 创建提示元素
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        color: white;
        padding: 12px 20px;
        border-radius: 25px;
        font-size: 14px;
        z-index: 10000;
        animation: slideIn 0.3s ease;
    `;
    toast.textContent = message;
    
    // 添加动画样式
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
    `;
    document.head.appendChild(style);
    
    document.body.appendChild(toast);
    
    // 3秒后自动移除
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(toast);
        }, 300);
    }, 3000);
}

// 更新导航栏活动状态
function updateNavigation() {
    const currentPage = window.location.pathname.split('/').pop();
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === currentPage || 
            (currentPage === '' && link.getAttribute('href') === 'index.html')) {
            link.classList.add('active');
        }
    });
}

// 平滑滚动到锚点
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// 添加页面加载动画
window.addEventListener('load', function() {
    document.body.style.opacity = '0';
    document.body.style.transition = 'opacity 0.5s ease';
    
    setTimeout(() => {
        document.body.style.opacity = '1';
    }, 100);
}); 