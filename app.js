import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.25; 
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; // Размер, на котором училась модель

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
        } catch (e) { 
            loadingMsg.innerText = "Ошибка загрузки best.onnx"; 
        }
    }
});

// --- 2. СНИМОК ---
btnSnap.addEventListener('click', async () => {
    if (!isModelReady) return;
    btnSnap.style.transform = "scale(0.9)";
    setTimeout(() => btnSnap.style.transform = "scale(1)", 150);

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

    setTimeout(async () => {
        try {
            const result = await runInference(ctx);
            loadingMsg.style.display = 'none';

            if (result && result.score > CONFIDENCE_THRESHOLD) {
                showResult(result.id);
            } else {
                if (result) {
                    const cardName = getCardName(result.id);
                    alert(`Не уверен. Это ${cardName}? (${(result.score * 100).toFixed(0)}%)\nПопробуй навести резкость.`);
                } else {
                    alert("Ничего не вижу. Включи свет.");
                }
            }
        } catch (e) {
            console.error(e);
            loadingMsg.style.display = 'none';
        }
    }, 50);
});

function getCardName(id) {
    const c = tarotDatabase.find(x => x.id === id);
    return c ? c.name : `ID ${id}`;
}

// --- 3. НЕЙРОСЕТЬ ---
async function runInference(ctx) {
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const data = imageData.data; 
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    const size = INPUT_SIZE * INPUT_SIZE;

    // R, G, B Планарно
    for (let i = 0; i < size; i++) {
        float32Data[i]          = data[i * 4]     / 255.0; // Red
        float32Data[i + size]   = data[i * 4 + 1] / 255.0; // Green
        float32Data[i + 2*size] = data[i * 4 + 2] / 255.0; // Blue
    }
    
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const results = await model.run({ images: inputTensor });
    const output = results[Object.keys(results)[0]]; // Получаем объект тензора целиком

    return parseYOLO_Adaptive(output);
}

// 🔥 АДАПТИВНЫЙ ПАРСЕР (САМ ОПРЕДЕЛЯЕТ РАЗМЕР) 🔥
function parseYOLO_Adaptive(tensor) {
    const dims = tensor.dims; // Например [1, 84, 33600]
    const data = tensor.data;
    
    // Автоматически берем количество якорей из модели
    const numAnchors = dims[2]; // Должно быть 33600
    const numClasses = dims[1] - 4; // 84 - 4 = 80
    
    console.log(`Model Geometry: ${numAnchors} anchors, ${numClasses} classes`);

    // Смещение: пропускаем 4 строки геометрии (4 * 33600)
    const geometryOffset = 4 * numAnchors;
    
    let maxScore = 0;
    let bestClassId = -1;

    // Проходим по всем предсказаниям
    for (let i = 0; i < numAnchors; i++) {
        for (let c = 0; c < numClasses; c++) {
            // Смещение геометрии + (Номер класса * шаг) + текущая колонка
            const idx = geometryOffset + (c * numAnchors) + i;
            const score = data[idx];

            if (score > maxScore) {
                maxScore = score;
                bestClassId = c;
            }
        }
    }

    console.log(`ZORKI: Best Class ${bestClassId}, Score ${maxScore}`);
    
    return { id: bestClassId, score: maxScore };
}

// --- 5. РЕЗУЛЬТАТ ---
function showResult(id) {
    const card = tarotDatabase.find(c => c.id === id);
    if (card) {
        const imgPath = card.img.includes('/') ? card.img : `./cards/${card.img}`;
        resultImg.src = imgPath;
        resultTitle.innerText = card.name;
        resultDesc.innerText = card.short;
        showScreen('result');
    }
}

btnBack.addEventListener('click', () => showScreen('start'));
btnReset.addEventListener('click', () => showScreen('camera'));
