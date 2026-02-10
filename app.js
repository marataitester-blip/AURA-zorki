import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.15; // Снизили до 15% (Сверхчувствительность)
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
        } catch (e) { loadingMsg.innerText = "Ошибка модели (404)"; }
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
    loadingMsg.innerText = "Анализ...";

    setTimeout(async () => {
        try {
            const result = await runInference(ctx);
            loadingMsg.style.display = 'none';

            if (result.found) {
                showResult(result.id);
            } else {
                // Если не нашли уверенно, говорим что почти нашли
                const cardName = getCardName(result.bestId);
                alert(`Не уверен. Похоже на: ${cardName} (Вероятность: ${(result.score * 100).toFixed(0)}%).\nПопробуй ближе.`);
            }
        } catch (e) {
            console.error(e);
            loadingMsg.style.display = 'none';
            alert("Ошибка вычислений: " + e.message);
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
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    for (let i = 0; i < float32Data.length; i++) {
        float32Data[i] = imageData.data[i * 4] / 255.0; 
    }
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const results = await model.run({ images: inputTensor });
    const output = results[Object.keys(results)[0]].data;

    return parseYOLO_Sensitive(output);
}

// 🔥 ЧУВСТВИТЕЛЬНЫЙ ПАРСЕР 🔥
function parseYOLO_Sensitive(data) {
    const numAnchors = 8400; 
    const numClasses = 80;
    
    let globalMaxScore = 0;
    let globalBestClass = -1;

    // Ищем максимум по всему массиву классов
    for (let i = 0; i < numAnchors; i++) {
        for (let c = 0; c < numClasses; c++) {
            // (4 строки геометрии пропускаем) + c
            const idx = (4 + c) * numAnchors + i;
            const score = data[idx];

            if (score > globalMaxScore) {
                globalMaxScore = score;
                globalBestClass = c;
            }
        }
    }

    console.log(`ZORKI: Best guess ${globalBestClass} (${globalMaxScore})`);

    // Возвращаем результат в любом случае, но ставим флаг found
    return {
        found: globalMaxScore > CONFIDENCE_THRESHOLD,
        id: globalBestClass,
        bestId: globalBestClass,
        score: globalMaxScore
    };
}

// --- 4. РЕЗУЛЬТАТ ---
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
