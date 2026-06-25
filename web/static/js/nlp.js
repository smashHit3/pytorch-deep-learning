const textInput = document.getElementById('text');
const modelSelect = document.getElementById('model');
const datasetSelect = document.getElementById('dataset');
const predictBtn = document.getElementById('predictBtn');
const loadingIndicator = document.getElementById('loading');
const errorBox = document.getElementById('error');
const resultBox = document.getElementById('result');
const predictionText = document.getElementById('prediction');
const confidenceText = document.getElementById('confidence');
const datasetNote = document.getElementById('dataset-note');
const resultModelBadge = document.getElementById('result-model');

if (predictBtn) {
    predictBtn.addEventListener('click', predict);
}

async function predict() {
    const text = textInput.value.trim();
    const model = modelSelect.value;
    const dataset = datasetSelect.value;

    resultBox.classList.remove('show', 'positive', 'negative', 'neutral');
    hideError();

    if (!text) {
        showError('Enter some text to classify.');
        return;
    }

    loadingIndicator.style.display = 'block';
    predictBtn.disabled = true;

    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('model_type', model);
        formData.append('dataset', dataset);

        const response = await fetch('/nlp/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (!response.ok) {
            showError(data.error || 'Prediction failed.');
            return;
        }

        showResult(data);
    } catch (error) {
        showError(`Network error: ${error.message}`);
    } finally {
        loadingIndicator.style.display = 'none';
        predictBtn.disabled = false;
    }
}

function showResult(data) {
    predictionText.textContent = data.prediction;
    confidenceText.textContent = `Confidence: ${data.confidence}`;
    resultModelBadge.textContent = data.model_type.toUpperCase();

    if (data.dataset === 'imdb') {
        resultBox.classList.add(data.prediction === 'Positive' ? 'positive' : 'negative');
        datasetNote.textContent = `Dataset: ${data.dataset_label}`;
    } else {
        resultBox.classList.add('neutral');
        datasetNote.textContent = `Dataset: ${data.dataset_label}`;
    }

    if (data.configured_dataset) {
        datasetNote.textContent += ' (served with the dataset saved in the model config)';
    }

    resultBox.classList.add('show');
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.add('show');
}

function hideError() {
    errorBox.textContent = '';
    errorBox.classList.remove('show');
}
