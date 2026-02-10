import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; 

const btnStart = document.getElementById('btn-start');
const btnSnap = document.getElementById('btn-snap');
const loadingMsg = document.getElementById('loading-msg');
const video = document.getElementById('camera-feed');
const screens = {
    start: document.getElementById('screen-start'),
    camera: document.getElementById('screen-camera'),
    result: document.getElementById('screen-result')
};

// Создаем панель для вывода цифр
const debugPanel = document.createElement('div');
debugPanel.style.position = 'absolute';
debugPanel.style.top = '10px';
debugPanel.style.left = '10px';
debugPanel.style.color = '#00ff00';
debugPanel.style.backgroundColor = 'rgba(0,0,0,0.8)';
debugPanel.style.padding = '10px';
debugPanel.style.fontFamily = 'monospace';
debugPanel.style.fontSize = '12px';
debugPanel.style.zIndex = '1000';
debugPanel.style.whiteSpace = 'pre-wrap';
debugPanel.innerHTML = "DIAGNOSTIC MODE<br>Waiting for snap...";
document.body.appendChild(debugPanel);

let model = null;

// 1. СТАРТ
btnStart.addEventListener('click', async () => {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens['camera'].classList.add('active');
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject = stream;
    } catch (e) { alert("Camera Error"); }

    if (!model) {
        loadingMsg.innerText = "Loading Model...";
        try {
            model = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
            });
            loadingMsg.style.display = 'none';
            btnSnap.disabled = false;
        } catch (e) {
            debugPanel.innerText = "Model Load Error: " + e.message;
        }
    }
});

// 2. СНИМОК И АНАЛИЗ
btnSnap.addEventListener('click', async () => {
    debugPanel.innerText = "Processing...";
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);

    try {
        const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
        const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
        for (let i = 0; i < float32Data.length; i++) {
            float32Data[i] = imageData.data[i * 4] / 255.0; // Normalization 0-1
        }
        const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

        const feeds = { images: inputTensor };
        const results = await model.run(feeds);
        const output = results[Object.keys(results)[0]];

        // --- ДИАГНОСТИКА ---
        const data = output.data;
        const dims = output.dims; // [1, 84, 8400] или [1, 8400, 84]?
        
        // Ищем самый большой сигнал в массиве
        let maxVal = -Infinity;
        let maxIdx = -1;
        for(let i=0; i<data.length; i++) {
            if(data[i] > maxVal) {
                maxVal = data[i];
                maxIdx = i;
            }
        }
        
        // Выводим отчет на экран
        debugPanel.innerHTML = `
        === DIAGNOSTIC REPORT ===
        DIMS: [${dims.join(', ')}]
        Total Size: ${data.length}
        
        MAX VALUE: ${maxVal.toFixed(4)}
        AT INDEX: ${maxIdx}
        
        SAMPLE (First 10):
        ${Array.from(data.slice(0, 10)).map(n => n.toFixed(4)).join(', ')}
        
        SAMPLE (Row Stride Test):
        Idx 8400: ${data[8400] ? data[8400].toFixed(4) : 'N/A'}
        `;

    } catch (e) {
        debugPanel.innerText = "Error: " + e.message;
    }
});
