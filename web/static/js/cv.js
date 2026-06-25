const fileInput = document.getElementById('file-input');
const uploadText = document.getElementById('upload-text');
const imagePreview = document.getElementById('image-preview');
const predictButton = document.getElementById('predict-btn');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result');
const errorSection = document.getElementById('error');
const topLabel = document.getElementById('top-label');
const topConfidence = document.getElementById('top-confidence');
const resultModel = document.getElementById('result-model');
const predictionList = document.getElementById('prediction-list');

if (fileInput && predictButton) {
    fileInput.addEventListener('change', previewImage);
    predictButton.addEventListener('click', uploadImage);
}

function previewImage() {
    clearError();
    resultSection.classList.remove('show');

    if (!fileInput.files || !fileInput.files[0]) {
        uploadText.textContent = 'Choose an image';
        imagePreview.style.display = 'none';
        predictButton.disabled = true;
        return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
        imagePreview.src = event.target.result;
        imagePreview.style.display = 'block';
        uploadText.textContent = fileInput.files[0].name;
        predictButton.disabled = false;
    };
    reader.readAsDataURL(fileInput.files[0]);
}

async function uploadImage() {
    if (!fileInput.files || !fileInput.files[0]) {
        showError('Choose an image before running inference.');
        return;
    }

    clearError();
    resultSection.classList.remove('show');
    predictButton.disabled = true;
    loading.style.display = 'block';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model_name', document.getElementById('model-select').value);

    try {
        const response = await fetch('/cv/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!response.ok) {
            showError(data.error || 'Image classification failed.');
            return;
        }

        renderPredictions(data);
    } catch (error) {
        showError(`Network error: ${error.message}`);
    } finally {
        loading.style.display = 'none';
        predictButton.disabled = false;
    }
}

function renderPredictions(data) {
    topLabel.textContent = data.top_label;
    topConfidence.textContent = `${data.top_confidence}% confidence`;
    resultModel.textContent = data.model_name;
    predictionList.innerHTML = '';

    data.predictions.forEach((prediction) => {
        const card = document.createElement('div');
        card.className = 'prediction-card';
        card.innerHTML = `
            <div class="prediction-row">
                <span>${prediction.label}</span>
                <span>${prediction.confidence}%</span>
            </div>
            <div class="confidence-track">
                <div class="confidence-fill" style="width: ${prediction.confidence}%"></div>
            </div>
        `;
        predictionList.appendChild(card);
    });

    resultSection.classList.add('show');
}

function showError(message) {
    errorSection.textContent = message;
    errorSection.classList.add('show');
}

function clearError() {
    errorSection.textContent = '';
    errorSection.classList.remove('show');
}
