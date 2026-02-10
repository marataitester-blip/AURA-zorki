import tarotDatabase from './tarot_db.js';

// --- КОНФИГУРАЦИЯ ---
const CONFIDENCE_THRESHOLD = 0.45; // Чуть снизили, но улучшили математику
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; 

// --- ЭЛЕМЕНТЫ UI ---
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

// --- 2. ЗАПУСК КАМЕРЫ И МОДЕЛИ ---
btnStart.addEventListener('click', async () => {
    showScreen('camera');
    
    // 1. Включаем камеру
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
        });
        video.srcObject = stream;
    } catch (e) {
        alert("Нет доступа к камере");
        return;
    }

    // 2. Грузим модель (если еще не загружена)
    if (!model) {
        try {
            model = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            isModelReady = true;
            loadingMsg.style.display = 'none';
            btnSnap.disabled = false;
        } catch (e) {
            loadingMsg.innerText = "Ошибка модели: " + e.message;
        }
    }
});

// --- 3. СНИМОК И АНАЛИЗ ---
btnSnap.addEventListener('click', async () => {
    if (!isModelReady) return;

    btnSnap.style.transform = "scale(0.8)";
    setTimeout(() => btnSnap.style.transform = "scale(1)", 100);

    // 1. Делаем "Фриз" картинки (Snap)
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const ctx = tempCanvas.getContext('2d');
    
    // Рисуем текущий кадр видео в квадрат 1280x1280 (растягиваем или кропаем)
    // Лучше скропать центр, чтобы не искажать пропорции карт
    const sourceMin = Math.min(video.videoWidth, video.videoHeight);
    const sx = (video.videoWidth - sourceMin) / 2;
    const sy = (video.videoHeight - sourceMin) / 2;
    ctx.drawImage(video, sx, sy, sourceMin, sourceMin, 0, 0, INPUT_SIZE, INPUT_SIZE);

    // 2. Анализируем
    const detection = await runInference(ctx);

    if (detection) {
        // Нашли карту!
        const cardData = tarotDatabase.find(c => c.id === detection.id);
        if (cardData) {
            // Показываем результат
            // Берем ЧИСТУЮ картинку из базы, а не фото (так красивее и понятнее пользователю)
            // Но если хочешь фото - можно использовать tempCanvas.toDataURL()
            resultImg.src = `./cards/${cardData.img}`; 
            resultTitle.innerText = cardData.name;
            resultDesc.innerText = cardData.short;
            
            showScreen('result');
        } else {
            alert(`ID ${detection.id} найден, но нет в базе.`);
        }
    } else {
        alert("Карта не распознана. Попробуй ближе или включи свет.");
    }
});

// --- 4. НЕЙРОСЕТЬ (ИСПРАВЛЕННАЯ ЛОГИКА) ---
async function runInference(ctx) {
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const inputTensor = preprocess(imageData.data, INPUT_SIZE, INPUT_SIZE);

    const feeds = { images: inputTensor };
    const results = await model.run(feeds);
    const output = results[Object.keys(results)[0]].data;

    return parseYOLOOutput_Fixed(output);
}

function preprocess(data, width, height) {
    const float32Data = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
        // Нормализация 0-255 -> 0.0-1.0
        float32Data[i] = data[i * 4] / 255.0;                   
        float32Data[i + width * height] = data[i * 4 + 1] / 255.0;       
        float32Data[i + 2 * width * height] = data[i * 4 + 2] / 255.0;   
    }
    return new ort.Tensor('float32', float32Data, [1, 3, width, height]);
}

// 🔥 ИСПРАВЛЕННЫЙ ПАРСЕР (YOLOv11 Output: [1, 84, 8400])
function parseYOLOOutput_Fixed(data) {
    const numClasses = 80;
    const numElements = 8400; // Количество "якорей" (predictions)
    
    // Структура данных: 84 строки (4 box + 80 classes), 8400 колонок
    // data[row * 8400 + col]
    
    let maxScore = 0;
    let bestClassId = -1;

    for (let i = 0; i < numElements; i++) {
        // Ищем максимальную уверенность среди всех классов для этого якоря
        let currentClassScore = 0;
        let currentClassId = -1;

        // Проходим по классам (начинаются с 4-й строки)
        for (let c = 0; c < numClasses; c++) {
            // Строка = 4 + c
            const score = data[(4 + c) * numElements + i];
            if (score > currentClassScore) {
                currentClassScore = score;
                currentClassId = c;
            }
        }

        if (currentClassScore > maxScore) {
            maxScore = currentClassScore;
            bestClassId = currentClassId;
        }
    }

    console.log(`Max Score: ${maxScore}, Class: ${bestClassId}`);

    if (maxScore > CONFIDENCE_THRESHOLD) {
        return { id: bestClassId, score: maxScore };
    }
    return null;
}

// Кнопки возврата
btnBack.addEventListener('click', () => showScreen('start'));
btnReset.addEventListener('click', () => showScreen('camera'));
