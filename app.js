import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.50; // Порог уверенности (50%)
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; // Размер, на котором училась модель

// --- ЭЛЕМЕНТЫ ИНТЕРФЕЙСА ---
const screens = {
    start: document.getElementById('screen-start'),
    camera: document.getElementById('screen-camera'),
    result: document.getElementById('screen-result')
};

// Кнопки
const btnStart = document.getElementById('btn-start');
const btnSnap = document.getElementById('btn-snap');
const btnBack = document.getElementById('btn-back-cam');
const btnReset = document.getElementById('btn-reset');

// Индикаторы
const loadingMsg = document.getElementById('loading-msg');
const video = document.getElementById('camera-feed');

// Элементы результата
const resultImg = document.getElementById('result-img');
const resultTitle = document.getElementById('result-title');
const resultDesc = document.getElementById('result-desc');

// Глобальные переменные
let model = null;
let isModelReady = false;

// --- 1. НАВИГАЦИЯ ПО ЭКРАНАМ ---
function showScreen(name) {
    // Скрываем все экраны
    Object.values(screens).forEach(s => s.classList.remove('active'));
    // Показываем нужный
    screens[name].classList.add('active');
}

// --- 2. ИНИЦИАЛИЗАЦИЯ (СТАРТ) ---
btnStart.addEventListener('click', async () => {
    showScreen('camera');
    
    // А. Запускаем камеру
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment', // Задняя камера
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        video.srcObject = stream;
    } catch (e) {
        alert("Ошибка: Не могу включить камеру. Разрешите доступ в браузере.");
        console.error(e);
        return;
    }

    // Б. Загружаем "мозг" (Нейросеть), если еще не загружен
    if (!model) {
        try {
            loadingMsg.innerText = "Загрузка Зрения...";
            loadingMsg.style.display = 'block';
            
            model = await ort.InferenceSession.create(MODEL_PATH, {
                executionProviders: ['wasm'], // WebAssembly (работает везде)
                graphOptimizationLevel: 'all'
            });
            
            isModelReady = true;
            loadingMsg.style.display = 'none';
            btnSnap.disabled = false; // Разблокируем кнопку спуска
            
            console.log("AURA ZORKI: Model Loaded");
        } catch (e) {
            loadingMsg.innerText = "Ошибка загрузки модели";
            console.error("Model Error:", e);
            alert("Не удалось загрузить файл best.onnx. Проверь, лежит ли он в корне GitHub.");
        }
    }
});

// --- 3. СЦЕНАРИЙ СЪЕМКИ (SNAP) ---
btnSnap.addEventListener('click', async () => {
    if (!isModelReady) return;

    // Эффект нажатия
    btnSnap.style.transform = "scale(0.8)";
    setTimeout(() => btnSnap.style.transform = "scale(1)", 150);

    // 1. Подготовка "холста" для нейросети
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const ctx = tempCanvas.getContext('2d');

    // 2. Берем кадр с видео. 
    // Важно: Кропаем центр (квадрат), чтобы не искажать геометрию карт
    const videoRatio = video.videoWidth / video.videoHeight;
    let sWidth, sHeight, sx, sy;

    if (videoRatio > 1) {
        // Горизонтальное видео: режем бока
        sHeight = video.videoHeight;
        sWidth = sHeight;
        sx = (video.videoWidth - sHeight) / 2;
        sy = 0;
    } else {
        // Вертикальное видео: режем верх/низ
        sWidth = video.videoWidth;
        sHeight = sWidth;
        sx = 0;
        sy = (video.videoHeight - sWidth) / 2;
    }

    // Рисуем квадратный кусок видео на канвас 1280x1280
    ctx.drawImage(video, sx, sy, sWidth, sHeight, 0, 0, INPUT_SIZE, INPUT_SIZE);

    // 3. Отправляем в нейросеть
    loadingMsg.style.display = 'block';
    loadingMsg.innerText = "Анализ...";
    
    try {
        const detection = await runInference(ctx);
        
        loadingMsg.style.display = 'none';

        if (detection) {
            // Успех! Показываем результат
            displayCardResult(detection.id);
        } else {
            alert("Карта не распознана. Попробуйте навести резкость или включить свет.");
        }
    } catch (e) {
        console.error(e);
        loadingMsg.style.display = 'none';
    }
});

// --- 4. МАТЕМАТИКА НЕЙРОСЕТИ (INFERENCE) ---
async function runInference(ctx) {
    // Получаем пиксели
    const imageData = ctx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    // Превращаем в тензор (формат для ИИ)
    const inputTensor = preprocess(imageData.data, INPUT_SIZE, INPUT_SIZE);

    // Запускаем
    const feeds = { images: inputTensor };
    const results = await model.run(feeds);
    
    // Получаем сырой ответ [1, 84, 8400]
    const output = results[Object.keys(results)[0]].data;

    // Расшифровываем
    return parseYOLOOutput_Correct(output);
}

// Преобразование картинки в цифры (0.0 - 1.0)
function preprocess(data, width, height) {
    const float32Data = new Float32Array(3 * width * height);
    for (let i = 0; i < width * height; i++) {
        float32Data[i] = data[i * 4] / 255.0;                   // R
        float32Data[i + width * height] = data[i * 4 + 1] / 255.0;       // G
        float32Data[i + 2 * width * height] = data[i * 4 + 2] / 255.0;   // B
    }
    return new ort.Tensor('float32', float32Data, [1, 3, width, height]);
}

// 🔥 ИСПРАВЛЕННЫЙ ПАРСЕР (УЧИТЫВАЕМ ГЕОМЕТРИЮ) 🔥
function parseYOLOOutput_Correct(data) {
    // Формат YOLOv11 output: [Batch, Channels, Anchors] -> [1, 84, 8400]
    // Строки 0-3: Геометрия (Center X, Center Y, Width, Height)
    // Строки 4-83: Классы (Вероятности для 0..79)
    
    const numAnchors = 8400; // Количество предсказаний
    const numClasses = 80;
    
    let maxScore = 0;
    let bestClassId = -1;

    // Пробегаем по всем 8400 возможным рамкам
    for (let i = 0; i < numAnchors; i++) {
        
        // Сначала ищем МАКСИМАЛЬНУЮ вероятность среди классов для этой рамки
        let currentClassMax = 0;
        let currentClassId = -1;

        // Цикл только по классам (пропускаем первые 4 строки геометрии!)
        for (let c = 0; c < numClasses; c++) {
            // Индекс в массиве = (номер_строки * ширина_строки) + номер_колонки
            // Строка классов начинается с 4
            const classRow = 4 + c;
            const score = data[classRow * numAnchors + i];

            if (score > currentClassMax) {
                currentClassMax = score;
                currentClassId = c;
            }
        }

        // Если эта рамка лучше предыдущей лучшей — запоминаем её
        if (currentClassMax > maxScore) {
            maxScore = currentClassMax;
            bestClassId = currentClassId;
            
            // ГЕОМЕТРИЯ (ДЛЯ ОТЛАДКИ)
            // Мы можем вытащить координаты, если захотим рисовать рамку
            // const x = data[0 * numAnchors + i];
            // const y = data[1 * numAnchors + i];
            // const w = data[2 * numAnchors + i];
            // const h = data[3 * numAnchors + i];
        }
    }

    console.log(`ZORKI SCAN: Class ${bestClassId} with confidence ${maxScore.toFixed(2)}`);

    if (maxScore > CONFIDENCE_THRESHOLD) {
        return { id: bestClassId, score: maxScore };
    }
    
    return null;
}

// --- 5. ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА ---
function displayCardResult(cardId) {
    // Ищем карту в базе
    const cardData = tarotDatabase.find(c => c.id === cardId);
    
    if (cardData) {
        // Заполняем экран результата
        // Используем цифровую картинку из папки cards
        resultImg.src = `./cards/${cardData.img}`; 
        
        resultTitle.innerText = cardData.name;
        resultDesc.innerText = cardData.short;
        
        // Переходим на экран
        showScreen('result');
    } else {
        alert("Ошибка базы данных: Карта найдена нейросетью, но отсутствует описание.");
    }
}

// --- 6. КНОПКИ УПРАВЛЕНИЯ ---
// Крестик на камере -> На старт
btnBack.addEventListener('click', () => {
    showScreen('start');
});

// Кнопка "Искать еще" -> Обратно на камеру
btnReset.addEventListener('click', () => {
    showScreen('camera');
    // Сбрасываем текст (косметика)
    resultTitle.innerText = "...";
});
