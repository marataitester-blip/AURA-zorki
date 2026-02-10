import tarotDatabase from './tarot_db.js';

// --- НАСТРОЙКИ ---
const CONFIDENCE_THRESHOLD = 0.60; // Порог уверенности (60%)
const MODEL_PATH = './best.onnx';
const INPUT_SIZE = 1280; // Размер, на котором училась модель

// --- СОСТОЯНИЕ ---
let model = null;
let video = document.getElementById('camera-feed');
let canvas = document.getElementById('detection-canvas');
let ctx = canvas.getContext('2d');
let scanBtn = document.getElementById('scan-btn');
let loadingIndicator = document.getElementById('loading-indicator');
let resultArea = document.getElementById('result-area');
let installBtn = document.getElementById('install-btn');

// --- 1. ИНИЦИАЛИЗАЦИЯ ---
async function init() {
    try {
        // Настройка камеры (Задняя камера предпочтительно)
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
        video.srcObject = stream;

        // Загрузка Нейросети
        loadingIndicator.innerText = "Загрузка Зрения...";
        loadingIndicator.style.display = 'block';
        
        // Создаем сессию ONNX (используем WebAssembly)
        model = await ort.InferenceSession.create(MODEL_PATH, {
            executionProviders: ['wasm'], 
            graphOptimizationLevel: 'all'
        });

        loadingIndicator.style.display = 'none';
        scanBtn.disabled = false;
        scanBtn.innerText = "СКАНИРОВАТЬ";
        
        // Подгоняем размер канваса под видео
        video.onloadedmetadata = () => {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        };

        console.log("AURA ZORKI: System Ready");

    } catch (e) {
        console.error(e);
        alert("Ошибка инициализации: " + e.message);
        loadingIndicator.innerText = "Ошибка доступа к камере или модели";
    }
}

// --- 2. ОБРАБОТКА КНОПКИ "СКАНИРОВАТЬ" ---
scanBtn.addEventListener('click', async () => {
    if (!model) return;

    scanBtn.innerText = "Анализ...";
    scanBtn.disabled = true;

    try {
        const detection = await runInference();
        
        if (detection) {
            displayResult(detection);
        } else {
            alert("Карта не распознана. Попробуйте поменять угол или свет.");
        }
    } catch (e) {
        console.error(e);
    } finally {
        scanBtn.innerText = "СКАНИРОВАТЬ";
        scanBtn.disabled = false;
    }
});

// --- 3. ЛОГИКА НЕЙРОСЕТИ (INFERENCE) ---
async function runInference() {
    // 1. Подготовка изображения (Preprocessing)
    // Рисуем текущий кадр видео на скрытый канвас 1280x1280
    const processCanvas = document.createElement('canvas');
    processCanvas.width = INPUT_SIZE;
    processCanvas.height = INPUT_SIZE;
    const pCtx = processCanvas.getContext('2d');
    pCtx.drawImage(video, 0, 0, INPUT_SIZE, INPUT_SIZE);
    
    const imageData = pCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const inputTensor = preprocess(imageData.data, INPUT_SIZE, INPUT_SIZE);

    // 2. Запуск модели
    const feeds = { images: inputTensor }; // Имя входа 'images' стандартно для YOLO
    const results = await model.run(feeds);
    
    // 3. Разбор результата (Postprocessing)
    // Выход YOLOv11 обычно [1, 84, 8400] -> (Batch, 4 box + 80 classes, Anchors)
    const output = results[Object.keys(results)[0]].data;
    
    return parseYOLOOutput(output);
}

// Преобразование картинки в Тензор [1, 3, 1280, 1280]
function preprocess(data, width, height) {
    const float32Data = new Float32Array(3 * width * height);
    
    // HWC -> NCHW и нормализация (0-255 -> 0.0-1.0)
    for (let i = 0; i < width * height; i++) {
        float32Data[i] = data[i * 4] / 255.0;                   // R
        float32Data[i + width * height] = data[i * 4 + 1] / 255.0;       // G
        float32Data[i + 2 * width * height] = data[i * 4 + 2] / 255.0;   // B
    }
    
    return new ort.Tensor('float32', float32Data, [1, 3, width, height]);
}

// Парсинг "спагетти" из цифр, которые выдает YOLO
function parseYOLOOutput(data) {
    const numAnchors = 8400; // Для размера 640 это 8400, для 1280 может быть больше, но структура [1, 84, N]
    const numClasses = 80;
    const stride = 4 + numClasses; // 84 строки
    
    let maxScore = 0;
    let bestClassId = -1;
    let bestBox = null;

    // Данные лежат плоско. Нам нужно пройтись по колонкам (якорям)
    // Структура: [x, y, w, h, class0_score, class1_score...] для каждого якоря
    // Но в ONNX Web данные часто транспонированы или идут подряд. 
    // Обычно YOLO export дает [Batch, Channel, Anchor].
    
    // Пробегаем по всем предсказаниям
    for (let i = 0; i < numAnchors; i++) {
        // Ищем максимальный класс для этого предсказания
        let currentMaxScore = 0;
        let currentClassId = -1;

        // Проверяем классы (начинаются с 4-го индекса)
        for (let c = 0; c < numClasses; c++) {
            // Формула индекса: (4 + c) * numAnchors + i
            // Потому что формат [Channel, Anchor]
            const score = data[(4 + c) * numAnchors + i];
            if (score > currentMaxScore) {
                currentMaxScore = score;
                currentClassId = c;
            }
        }

        if (currentMaxScore > maxScore) {
            maxScore = currentMaxScore;
            bestClassId = currentClassId;
            // Координаты (нам они сейчас не важны для UI, но нужны для проверки)
            // x = data[0 * numAnchors + i]
            // y = data[1 * numAnchors + i] ...
        }
    }

    if (maxScore > CONFIDENCE_THRESHOLD) {
        return {
            id: bestClassId,
            score: maxScore
        };
    }
    
    return null;
}

// --- 4. ОТОБРАЖЕНИЕ РЕЗУЛЬТАТА ---
function displayResult(detection) {
    const cardData = tarotDatabase.find(c => c.id === detection.id);
    
    if (cardData) {
        resultArea.style.display = 'block';
        document.getElementById('card-name').innerText = cardData.name;
        document.getElementById('card-image').src = cardData.image;
        document.getElementById('card-short').innerText = cardData.short;
        
        // Плавный скролл к результату
        resultArea.scrollIntoView({ behavior: 'smooth' });
    } else {
        alert(`Класс ${detection.id} найден, но нет в базе данных.`);
    }
}

// --- 5. ЛОГИКА PWA (УСТАНОВКА) ---
let deferredPrompt;
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredPrompt = e;
    installBtn.style.display = 'block';
});

installBtn.addEventListener('click', async () => {
    if (deferredPrompt) {
        deferredPrompt.prompt();
        const { outcome } = await deferredPrompt.userChoice;
        if (outcome === 'accepted') {
            installBtn.style.display = 'none';
        }
        deferredPrompt = null;
    }
});

// Запуск при загрузке страницы
init();
