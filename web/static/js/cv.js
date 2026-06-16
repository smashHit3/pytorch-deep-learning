function previewImage(input) {
    const preview = document.getElementById('image-preview');
    const text = document.getElementById('upload-text');
    const btn = document.getElementById('predict-btn');

    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
            text.style.display = 'none';
            btn.disabled = false;
        }
        reader.readAsDataURL(input.files[0]);
    }
}

async function uploadImage() {
    const fileInput = document.getElementById('file-input');
    const modelSelect = document.getElementById('model-select');
    const btn = document.getElementById('predict-btn');
    const loader = document.getElementById('loading');
    const resultDiv = document.getElementById('result');
    const topLabel = document.getElementById('top-label');
    const predList = document.getElementById('prediction-list');

    if (!fileInput.files[0]) return;

    btn.disabled = true;
    loader.style.display = 'block';
    resultDiv.classList.remove('show');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('model_name', modelSelect.value);

    try {
        const response = await fetch('/cv/predict', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) {
            alert('Error: ' + data.error);
        } else {
            topLabel.innerText = `Prediction: ${data.top_label}`;

            predList.innerHTML = '';
            data.predictions.forEach(pred => {
                const div = document.createElement('div');
                div.className = 'prediction-item';
                div.innerHTML = `
                    <span>${pred.label}</span>
                    <span>${pred.confidence}%</span>
                `;

                const barContainer = document.createElement('div');
                barContainer.className = 'confidence-bar';
                barContainer.innerHTML = `<div class="confidence-fill" style="width: ${pred.confidence}%"></div>`;

                predList.appendChild(div);
                predList.appendChild(barContainer);
            });

            resultDiv.classList.add('show');
        }
    } catch (e) {
        alert('Network error occurred');
    } finally {
        btn.disabled = false;
        loader.style.display = 'none';
    }
}
