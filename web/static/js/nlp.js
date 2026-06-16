async function predict() {
    const text = document.getElementById('text').value.trim();
    const model = document.getElementById('model').value;

    document.getElementById('result').classList.remove('show');
    document.getElementById('error').style.display = 'none';

    if (!text) {
        showError('Please enter some text to classify');
        return;
    }

    document.getElementById('loading').style.display = 'block';
    document.getElementById('predictBtn').disabled = true;

    try {
        const formData = new FormData();
        formData.append('text', text);
        formData.append('model_type', model);

        const response = await fetch('/nlp/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            showResult(data);
        } else {
            showError(data.error || 'Prediction failed');
        }
    } catch (err) {
        showError('Network error: ' + err.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('predictBtn').disabled = false;
    }
}

function showResult(data) {
    const resultEl = document.getElementById('result');
    const predictionEl = document.getElementById('prediction');
    const confidenceEl = document.getElementById('confidence');

    predictionEl.textContent = data.prediction;
    confidenceEl.textContent = `Confidence: ${data.confidence}`;

    resultEl.classList.remove('positive', 'negative');
    if (data.prediction === 'Positive') {
        resultEl.classList.add('positive');
    } else {
        resultEl.classList.add('negative');
    }

    resultEl.classList.add('show');
}

function showError(message) {
    const errorEl = document.getElementById('error');
    errorEl.textContent = message;
    errorEl.style.display = 'block';
}
