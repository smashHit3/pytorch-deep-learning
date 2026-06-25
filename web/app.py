import sys
import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import logic from our inference module
from cv_sources.classification.inference import (
    load_model,
    run_inference,
    LABELS,
    DEFAULT_MEAN,
    DEFAULT_STD,
    DEFAULT_RESIZE,
    DEFAULT_CROP
)
from cv_sources.data_processor import dogs_vs_cats, fashion_mnist
from cv_sources.classification.train import MODEL_FILE_MAP

# NLP imports - use the improved NLPInferenceEngine
from nlp_sources.inference import NLPInferenceEngine, get_class_names
from nlp_sources.models import lstm, gru, transformer

app = FastAPI()

# Mount static files for CSS
STATIC_DIR = PROJECT_ROOT / "web" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates for serving the HTML page
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------- Global State --------------------------
# Configuration Defaults
DEFAULT_MODEL = "alexnet"
DEFAULT_DATASET = dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Cache: stores loaded models as {model_name: model_object}
model_cache: Dict[str, torch.nn.Module] = {}

# NLP Model Cache: stores NLPInferenceEngine instances
nlp_cache: Dict[str, NLPInferenceEngine] = {}

# -------------------------- Helper Functions --------------------------
def get_model(model_name: str):
    """Load model from weights file and cache it to avoid redundant loading"""
    if model_name in model_cache:
        return model_cache[model_name]

    print(f"🚀 Loading model {model_name} into memory...")
    try:
        # Get weight filename from the shared map in train.py
        weight_filename = MODEL_FILE_MAP.get(model_name, "model.pth")
        weight_path = PROJECT_ROOT / "cv_sources" / "results" / weight_filename

        if not weight_path.exists():
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        # num_classes = 2 for dogs_vs_cats
        num_classes = 2 if DEFAULT_DATASET == dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS else 10

        model = load_model(
            model_name=model_name,
            weight_path=weight_path,
            num_classes=num_classes,
            device=DEVICE
        )
        model_cache[model_name] = model
        print(f"✅ Model {model_name} loaded and cached!")
        return model
    except Exception as e:
        print(f"❌ Error loading model {model_name}: {e}")
        return None

def get_transforms(dataset_name: str):
    """Return the transformation pipeline based on the dataset (Mirrors inference.py)"""
    from torchvision import transforms

    if dataset_name == fashion_mnist.DATASET_NAME_FASHION_MNIST:
        return transforms.Compose([
            transforms.Resize((DEFAULT_CROP, DEFAULT_CROP)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ])
    elif dataset_name == dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS:
        return transforms.Compose([
            transforms.Resize((DEFAULT_RESIZE, DEFAULT_RESIZE)),
            transforms.CenterCrop(DEFAULT_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((DEFAULT_RESIZE, DEFAULT_RESIZE)),
            transforms.CenterCrop(DEFAULT_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD)
        ])


# -------------------------- NLP Helper Functions --------------------------
NLP_RESULTS_DIR = PROJECT_ROOT / "nlp_sources" / "results"

def get_nlp_model(model_type: str, dataset: str = "imdb", embedding_dim: int = 128, hidden_dim: int = 256):
    """Load NLP model using the improved NLPInferenceEngine"""
    cache_key = (model_type, dataset, embedding_dim, hidden_dim)
    if cache_key in nlp_cache:
        return nlp_cache[cache_key]

    # Load vocabulary from file (no dataset loading needed)
    from nlp_sources.data_processor.base import Vocabulary
    from nlp_sources.data_processor import text_data
    
    vocab_path = NLP_RESULTS_DIR / f"vocab_{dataset}.json"
    if vocab_path.exists():
        vocab = Vocabulary.load(str(vocab_path))
    else:
        # Fallback: load from dataset if vocab file doesn't exist
        logger.info(f"Vocab file not found at {vocab_path}, loading from dataset...")
        _, _, vocab, num_classes = text_data.load_data(dataset, batch_size=32, max_seq_len=256)
    num_classes = 2 if dataset == "imdb" else 4  # AG_NEWS has 4 classes

    # Build model using NLPInferenceEngine
    model_path = NLP_RESULTS_DIR / f"{model_type}.pth"
    
    try:
        engine = NLPInferenceEngine(
            model_type=model_type,
            model_path=model_path,
            vocab=vocab,
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fp16=False  # Disable FP16 for web inference by default
        )
        nlp_cache[cache_key] = engine
        logger.info(f"NLP model {model_type} loaded and cached!")
        return engine
    except Exception as e:
        logger.error(f"Failed to load NLP model {model_type}: {e}")
        raise


def nlp_predict(engine: NLPInferenceEngine, text: str, max_seq_len: int = 256):
    """Predict sentiment for text using NLPInferenceEngine"""
    preds, confidences = engine.predict_batch([text], max_seq_len)
    return preds[0], confidences[0][preds[0]]

# -------------------------- Endpoints --------------------------

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the landing page"""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/cv", response_class=HTMLResponse)
async def cv_page(request: Request):
    """Serve the CV classification page"""
    available_models = list(MODEL_FILE_MAP.keys())
    return templates.TemplateResponse(
        request=request,
        name="cv.html",
        context={
            "model_name": DEFAULT_MODEL,
            "available_models": available_models
        }
    )


@app.post("/cv/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...)
):
    """Handle image upload and run inference using the selected model"""
    # 1. Get or load the selected model
    model = get_model(model_name)
    if model is None:
        return {"error": f"Model '{model_name}' could not be loaded. Check server logs."}

    try:
        # 2. Read image bytes and convert to PIL
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 3. Preprocess using the dataset-specific pipeline
        preprocess = get_transforms(DEFAULT_DATASET)
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        # 4. Run inference
        top_probs, top_indices = run_inference(model, img_tensor, top_k=2)

        # 5. Map indices to labels
        dataset_labels = LABELS.get(DEFAULT_DATASET, ["Class 0", "Class 1"])
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = dataset_labels[idx] if idx < len(dataset_labels) else f"Class {idx}"
            results.append({"label": label, "confidence": round(prob * 100, 2)})

        return {
            "predictions": results,
            "top_label": results[0]["label"],
            "top_confidence": results[0]["confidence"]
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------- NLP Endpoints --------------------------

@app.get("/nlp", response_class=HTMLResponse)
async def nlp_page(request: Request):
    """Serve the NLP classification page"""
    return templates.TemplateResponse(
        request=request,
        name="nlp.html",
        context={}
    )


@app.post("/nlp/predict")
async def nlp_predict_endpoint(
    text: str = Form(...),
    model_type: str = Form("lstm"),
    dataset: str = Form("imdb")
):
    """Handle text classification request using NLPInferenceEngine"""
    if not text or not text.strip():
        return {"error": "No text provided"}

    try:
        engine = get_nlp_model(model_type, dataset)
        pred, confidence = nlp_predict(engine, text)

        # Get class names based on dataset
        class_names = get_class_names(dataset)

        return {
            "prediction": class_names[pred],
            "class_id": pred,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "dataset": dataset
        }
    except Exception as e:
        logger.error(f"NLP prediction failed: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    # Run the app on localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
