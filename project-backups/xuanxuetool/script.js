// 全局变量
let isTyping = false;

// 梅花易数数据
const plumBlossomData = {
    // 八卦
    trigrams: [
        { name: "乾", symbol: "☰", element: "金", nature: "天", meaning: "刚健、领导、父亲" },
        { name: "坤", symbol: "☷", element: "土", nature: "地", meaning: "柔顺、包容、母亲" },
        { name: "震", symbol: "☳", element: "木", nature: "雷", meaning: "动、变化、长子" },
        { name: "巽", symbol: "☴", element: "木", nature: "风", meaning: "入、渗透、长女" },
        { name: "坎", symbol: "☵", element: "水", nature: "水", meaning: "险、智慧、中男" },
        { name: "离", symbol: "☲", element: "火", nature: "火", meaning: "明、文明、中女" },
        { name: "艮", symbol: "☶", element: "土", nature: "山", meaning: "止、稳重、少男" },
        { name: "兑", symbol: "☱", element: "金", nature: "泽", meaning: "悦、口舌、少女" }
    ],
    
    // 五行
    elements: ["金", "木", "水", "火", "土"],
    
    // 运势等级
    fortuneLevels: ["大吉", "中吉", "小吉", "平", "小凶", "中凶", "大凶"],
    
    // 方位
    directions: ["东", "南", "西", "北", "东南", "西南", "西北", "东北"],
    
    // 颜色
    colors: ["白色", "绿色", "黑色", "红色", "黄色", "蓝色", "紫色", "橙色"]
};

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// 初始化应用
function initializeApp() {
    // 设置默认日期为今天
    const today = new Date().toISOString().split('T')[0];
    const birthDateInput = document.getElementById('birthDate');
    if (birthDateInput) {
        birthDateInput.value = today;
    }
}

// 测算今日运势
function calculateFortune() {
    const birthDateInput = document.getElementById('birthDate');
    const calculateBtn = document.getElementById('calculateBtn');
    
    if (!birthDateInput.value) {
        alert('请选择出生日期');
        return;
    }
    
    if (isTyping) return;
    
    // 禁用按钮
    if (calculateBtn) {
        calculateBtn.disabled = true;
    }
    
    // 显示AI正在思考
    showTypingIndicator();
    
    // 模拟AI思考时间
    setTimeout(() => {
        hideTypingIndicator();
        const response = generateFortuneResponse(birthDateInput.value);
        addMessage(response, 'ai');
        
        // 重新启用按钮
        if (calculateBtn) {
            calculateBtn.disabled = false;
        }
    }, 2000 + Math.random() * 1000);
}

// 添加消息到聊天界面
function addMessage(content, sender) {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = sender === 'ai' ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = content.replace(/\n/g, '<br>');
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 显示打字指示器
function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.className = 'message ai-message typing-message';
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content typing-indicator">
            AI正在思考
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    isTyping = true;
}

// 隐藏打字指示器
function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
    isTyping = false;
}

// 生成梅花易数运势响应
function generateFortuneResponse(birthDate) {
    const date = new Date(birthDate);
    const month = date.getMonth() + 1;
    const day = date.getDate();
    const hour = new Date().getHours();
    
    // 梅花易数起卦
    const upperTrigram = plumBlossomData.trigrams[Math.floor(Math.random() * 8)];
    const lowerTrigram = plumBlossomData.trigrams[Math.floor(Math.random() * 8)];
    const changingLine = Math.floor(Math.random() * 6) + 1;
    
    // 生成运势
    const overallFortune = plumBlossomData.fortuneLevels[Math.floor(Math.random() * 7)];
    const loveFortune = plumBlossomData.fortuneLevels[Math.floor(Math.random() * 7)];
    const careerFortune = plumBlossomData.fortuneLevels[Math.floor(Math.random() * 7)];
    const wealthFortune = plumBlossomData.fortuneLevels[Math.floor(Math.random() * 7)];
    const healthFortune = plumBlossomData.fortuneLevels[Math.floor(Math.random() * 7)];
    
    // 幸运元素
    const luckyElement = plumBlossomData.elements[Math.floor(Math.random() * 5)];
    const luckyDirection = plumBlossomData.directions[Math.floor(Math.random() * 8)];
    const luckyColor = plumBlossomData.colors[Math.floor(Math.random() * 8)];
    
    return `🌸 <strong>梅花易数今日运势</strong><br><br>
    <strong>🔮 起卦信息：</strong><br>
    • 上卦：${upperTrigram.name} ${upperTrigram.symbol} (${upperTrigram.nature})<br>
    • 下卦：${lowerTrigram.name} ${lowerTrigram.symbol} (${lowerTrigram.nature})<br>
    • 变爻：第${changingLine}爻<br><br>
    
    <strong>📊 今日运势总览：</strong><br>
    • 综合运势：${overallFortune}<br>
    • 感情运势：${loveFortune}<br>
    • 事业运势：${careerFortune}<br>
    • 财运状况：${wealthFortune}<br>
    • 健康运势：${healthFortune}<br><br>
    
    <strong>🌸 梅花易数解析：</strong><br>
    ${getPlumBlossomInterpretation(upperTrigram, lowerTrigram, changingLine)}<br><br>
    
    <strong>🌟 今日宜忌：</strong><br>
    • 幸运方位：${luckyDirection}<br>
    • 幸运颜色：${luckyColor}<br>
    • 幸运元素：${luckyElement}<br>
    • 幸运时辰：${getLuckyHour(hour)}<br><br>
    
    <strong>💫 梅花易数建议：</strong><br>
    ${getPlumBlossomAdvice(overallFortune, upperTrigram, lowerTrigram)}`;
}

// 梅花易数解析
function getPlumBlossomInterpretation(upperTrigram, lowerTrigram, changingLine) {
    const interpretations = [
        `上卦${upperTrigram.name}代表${upperTrigram.meaning}，下卦${lowerTrigram.name}代表${lowerTrigram.meaning}。`,
        `第${changingLine}爻变化，预示着${getChangingLineMeaning(changingLine)}。`,
        `此卦象显示今日${getTrigramCombination(upperTrigram, lowerTrigram)}，`,
        `建议${getTrigramAdvice(upperTrigram, lowerTrigram)}。`
    ];
    
    return interpretations.join(' ');
}

// 变爻含义
function getChangingLineMeaning(line) {
    const meanings = [
        "初爻变，基础稳固，适合开始新计划",
        "二爻变，人际关系重要，注意沟通",
        "三爻变，内心变化，需要调整心态",
        "四爻变，外部环境变化，需要适应",
        "五爻变，领导地位，需要承担责任",
        "上爻变，达到顶点，需要谨慎行事"
    ];
    return meanings[line - 1];
}

// 卦象组合解析
function getTrigramCombination(upper, lower) {
    const combinations = [
        "天地交泰，阴阳调和，运势平稳",
        "雷风相薄，变化迅速，需要灵活应对",
        "水火既济，矛盾统一，需要平衡",
        "山泽通气，内外和谐，运势良好",
        "天雷无妄，诚实守信，避免妄为",
        "地火明夷，光明被掩，需要耐心",
        "风山渐，循序渐进，稳步发展",
        "水天需，等待时机，不可急躁"
    ];
    return combinations[Math.floor(Math.random() * combinations.length)];
}

// 卦象建议
function getTrigramAdvice(upper, lower) {
    const advice = [
        "保持内心平静，顺应自然规律",
        "发挥主动性，但不可过于急躁",
        "注重人际关系，寻求他人帮助",
        "保持谦逊态度，学习他人长处",
        "坚持原则，但要有灵活性",
        "保持耐心，等待合适时机",
        "发挥创造力，寻找新的机会",
        "保持平衡，避免极端行为"
    ];
    return advice[Math.floor(Math.random() * advice.length)];
}

// 获取幸运时辰
function getLuckyHour(hour) {
    const timeSlots = [
        "子时(23-1点)", "丑时(1-3点)", "寅时(3-5点)", "卯时(5-7点)",
        "辰时(7-9点)", "巳时(9-11点)", "午时(11-13点)", "未时(13-15点)",
        "申时(15-17点)", "酉时(17-19点)", "戌时(19-21点)", "亥时(21-23点)"
    ];
    return timeSlots[Math.floor(Math.random() * timeSlots.length)];
}

// 梅花易数建议
function getPlumBlossomAdvice(fortune, upper, lower) {
    const adviceMap = {
        "大吉": "今日运势极佳，适合重要决策和行动，把握机会，但不可骄傲自满。",
        "中吉": "运势良好，可以积极行动，但需要保持谨慎，避免过度自信。",
        "小吉": "运势平稳，适合稳步发展，保持积极心态，会有小收获。",
        "平": "运势平稳，保持现状，不宜大动作，适合思考和规划。",
        "小凶": "运势不佳，需要谨慎行事，避免重要决策，保持低调。",
        "中凶": "运势较差，需要特别小心，避免冲突，保持耐心等待。",
        "大凶": "运势极差，不宜重要行动，保持低调，等待时机转变。"
    };
    
    return adviceMap[fortune] || adviceMap["平"];
}

