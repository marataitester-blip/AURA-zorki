import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.40; // 40%
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; 

// --- ЭЛЕМЕНТЫ ---
const screens = {
    start: document.getElementById('screen-start'),
    camera: document.getElementById('screen-camera'),
    result: document.getElementById('screen-result')
};

const btnStart = document.getElementById('btn-start');
const btnSnap = document.getElementById('btn-snap');
const btnBack = document.getElementById('btn-back-cam');
const btnReset = document.getElementById('btn-reset');
const loadingMsg = document.getElementById('loading-msg');
const video = document.getElementById('camera-feed');
const resultImg = document.getElementById('result-img');
const resultTitle = document.getElementById('result-title');
const resultDesc = document.getElementById('result-desc');

let model = null;
let isModelReady = false;

// --- 1. СТАРТ ---
function showScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[name].classList.add('active');
}

btnStart.addEventListener('click', async () => {
    showScreen('camera');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject = stream;
    } catch (e) { alert("Camera error"); }

    if (!model) {
        loadingMsg.innerText = "Греем Нейроны...";
        try {
            model = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            isModelReady = true;
            loadingMsg.style.display = 'none';
            btnSnap.disabled = false;
        } catch (e) { loadingMsg.innerText = "Ошибка модели"; }
    }
});

// --- 2. СНИМОК ---
btnSnap.addEventListener('click', async () => {
    if (!isModelReady) return;
    btnSnap.style.transform = "scale(0.9)";
    setTimeout(() => btnSnap.style.transform = "scale(1)", 150);

    // Подготовка канваса
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const ctx = tempCanvas.getContext('2d');
    
    // Кроп центра
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - minDim) / 2;
    const sy = (video.videoHeight - minDim) / 2;
    ctx.drawImage(video, sx, sy, minDim, minDim, 0, 0, INPUT_SIZE, INPUT_SIZE);

    loadingMsg.style.display = 'block';
    loadingMsg.innerText = "Смотрю...";

    // Пауза чтобы UI обновился
    setTimeout(async () => {
        try {
            const detection = await runInference(ctx);
            loadingMsg.style.display = 'none';

            if (detection) {
                showResult(detection.id);
            } else {
                alert("Ничего не вижу. Попробуй светлее.");
            }
        } catch (e) {
            console.error(e);
            loadingMsg.style.display = 'none';
        }
    }, 50);
});

// --- 3. НЕЙРОСЕТЬ ---
async function runInference(ctx) {
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // Нормализация
    for (let i = 0; i < float32Data.length; i++) {
        float32Data[i] = imageData.data[i * 4] / 255.0; 
    }
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const results = await model.run({ images: inputTensor });
    const output = results[Object.keys(results)[0]].data;

    return parseYOLO_Brutal(output);
}

// 🔥 ЖЕСТКИЙ ПАРСЕР (БЕЗ ГЕОМЕТРИИ) 🔥
function parseYOLO_Brutal(data) {
    const numAnchors = 8400; 
    const numClasses = 80;
    
    // СМЕЩЕНИЕ: Пропускаем первые 4 строки (4 * 8400 элементов)
    // Это X, Y, W, H. Мы их просто игнорируем.
    const startOffset = 4 * numAnchors;
    
    let maxScore = 0;
    let bestClassId = -1;

    // Проходим по всем "столбикам" (якорям)
    for (let i = 0; i < numAnchors; i++) {
        
        // Внутри каждого якоря ищем победивший класс
        for (let c = 0; c < numClasses; c++) {
            
            // Индекс = (Смещение_классов + Номер_класса) * Ширина + Текущий_якорь
            // Но в плоском массиве [Batch, Channel, Anchor] это:
            // (4 + c) * 8400 + i
            
            const idx = (4 + c) * numAnchors + i;
            const score = data[idx];

            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }
    }

    console.log(`Max Score found: ${maxScore} for Class: ${bestClassId}`);

    // Если "Императрица" (ID 3) имеет score > 1.0, значит мы все еще читаем геометрию.
    // Но с этим кодом это невозможно.
    
    if (maxScore > CONFIDENCE_THRESHOLD) {
        return { id: bestClassId, score: maxScore };
    }
    return null;
}

// --- 4. ПОКАЗАТЬ РЕЗУЛЬТАТ ---
function showResult(id) {
    const card = tarotDatabase.find(c => c.id === id);
    if (card) {
        // Если image путь не содержит слэш, добавляем папку
        const imgPath = card.img.includes('/') ? card.img : `./cards/${card.img}`;
        
        resultImg.src = imgPath;
        resultTitle.innerText = card.name;
        resultDesc.innerText = card.short;
        showScreen('result');
    }
}

btnBack.addEventListener('click', () => showScreen('start'));
btnReset.addEventListener('click', () => showScreen('camera'));
