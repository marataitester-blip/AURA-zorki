import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.50; // 50% уверенности
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

// --- 1. НАВИГАЦИЯ ---
function showScreen(name) {
    Object.values(screens).forEach(s => s.classList.remove('active'));
    screens[name].classList.add('active');
}

// --- 2. СТАРТ ---
btnStart.addEventListener('click', async () => {
    showScreen('camera');
    
    // Камера
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject = stream;
    } catch (e) {
        alert("Нет доступа к камере. Проверь настройки браузера.");
    }

    // Модель
    if (!model) {
        try {
            loadingMsg.innerText = "Загрузка Зрения...";
            model = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            isModelReady = true;
            loadingMsg.style.display = 'none';
            btnSnap.disabled = false;
        } catch (e) {
            loadingMsg.innerText = "Ошибка: best.onnx не найден или битый.";
        }
    }
});

// --- 3. СЪЕМКА ---
btnSnap.addEventListener('click', async () => {
    if (!isModelReady) return;

    // Анимация кнопки
    btnSnap.style.transform = "scale(0.8)";
    setTimeout(() => btnSnap.style.transform = "scale(1)", 150);

    // Подготовка
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const ctx = tempCanvas.getContext('2d');

    // Кроп центра (чтобы не искажать пропорции)
    const minDim = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - minDim) / 2;
    const sy = (video.videoHeight - minDim) / 2;
    ctx.drawImage(video, sx, sy, minDim, minDim, 0, 0, INPUT_SIZE, INPUT_SIZE);

    // Анализ
    loadingMsg.style.display = 'block';
    loadingMsg.innerText = "Анализ...";

    try {
        const detection = await runInference(ctx);
        loadingMsg.style.display = 'none';

        if (detection) {
            showResult(detection.id);
        } else {
            alert("Карта не найдена. Попробуй ближе или включи свет.");
        }
    } catch (e) {
        console.error(e);
        loadingMsg.style.display = 'none';
    }
});

// --- 4. МОЗГИ (ИСПРАВЛЕННАЯ МАТЕМАТИКА) ---
async function runInference(ctx) {
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const float32Data = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
    
    // HWC -> NCHW Normalization
    for (let i = 0; i < float32Data.length / 3; i++) {
        float32Data[i] = imageData.data[i * 4] / 255.0;                   // R
        float32Data[i + INPUT_SIZE**2] = imageData.data[i * 4 + 1] / 255.0; // G
        float32Data[i + 2 * INPUT_SIZE**2] = imageData.data[i * 4 + 2] / 255.0; // B
    }
    const inputTensor = new ort.Tensor('float32', float32Data, [1, 3, INPUT_SIZE, INPUT_SIZE]);

    const results = await model.run({ images: inputTensor });
    const output = results[Object.keys(results)[0]].data; // Сырой массив

    return parseYOLO_Correct(output);
}

// 🔥 ГЛАВНОЕ ИСПРАВЛЕНИЕ 🔥
function parseYOLO_Correct(data) {
    const numAnchors = 8400; // Количество колонок
    const numClasses = 80;   // Количество классов
    
    // Структура массива [1, 84, 8400]:
    // Первые 8400 чисел = Center X
    // Следующие 8400 = Center Y
    // Следующие 8400 = Width
    // Следующие 8400 = Height (ВОТ ОНА, ИМПЕРАТРИЦА!)
    // Следующие 8400 = Class 0 Score
    // ...
    
    let maxScore = 0;
    let bestClassId = -1;

    // Мы бежим по колонкам (Anchor 0 -> 8399)
    for (let i = 0; i < numAnchors; i++) {
        
        let currentClassMax = 0;
        let currentClassId = -1;

        // Проверяем классы (начинаются со смещения 4 * 8400)
        for (let c = 0; c < numClasses; c++) {
            // Формула доступа к ячейке:
            // (Номер_Свойства * 8400) + Номер_Якоря
            const propertyIndex = 4 + c; 
            const value = data[propertyIndex * numAnchors + i];

            if (value > currentClassMax) {
                currentClassMax = value;
                currentClassId = c;
            }
        }

        if (currentClassMax > maxScore) {
            maxScore = currentClassMax;
            bestClassId = currentClassId;
        }
    }

    console.log(`ZORKI: Found Class ${bestClassId} with score ${maxScore}`);

    if (maxScore > CONFIDENCE_THRESHOLD) {
        return { id: bestClassId, score: maxScore };
    }
    return null;
}

// --- 5. РЕЗУЛЬТАТ ---
function showResult(id) {
    const card = tarotDatabase.find(c => c.id === id);
    if (card) {
        // Добавляем префикс cards/ если его нет
        const imgPath = card.img.includes('/') ? card.img : `./cards/${card.img}`;
        
        resultImg.src = imgPath;
        resultTitle.innerText = card.name;
        resultDesc.innerText = card.short;
        showScreen('result');
    }
}

// Кнопки Назад
btnBack.addEventListener('click', () => showScreen('start'));
btnReset.addEventListener('click', () => {
    showScreen('camera');
    resultTitle.innerText = "...";
});
