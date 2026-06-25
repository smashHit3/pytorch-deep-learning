import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to system path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError

from cv_sources.classification.inference import (
    DEFAULT_CROP,
    DEFAULT_MEAN,
    DEFAULT_RESIZE,
    DEFAULT_STD,
    LABELS,
    load_model,
    run_inference,
)
from cv_sources.classification.train import MODEL_FILE_MAP
from cv_sources.data_processor import dogs_vs_cats, fashion_mnist
from nlp_sources.data_processor import text_data
from nlp_sources.inference import NLPInferenceEngine, get_class_names, load_model_config
from nlp_sources.train import MODEL_FILE_MAP as NLP_MODEL_FILE_MAP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files for CSS
STATIC_DIR = PROJECT_ROOT / "web" / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Setup templates for serving the HTML page
TEMPLATES_DIR = PROJECT_ROOT / "web" / "templates"
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# -------------------------- Global State --------------------------
DEFAULT_MODEL = "alexnet"
DEFAULT_DATASET = dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS
DEFAULT_TEXT_DATASET = text_data.DATASET_NAME_IMDB
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CV_RESULTS_DIR = PROJECT_ROOT / "cv_sources" / "results"
NLP_RESULTS_DIR = PROJECT_ROOT / "nlp_sources" / "results"

TEXT_DATASETS: Dict[str, str] = {
    text_data.DATASET_NAME_IMDB: "IMDB Reviews",
    text_data.DATASET_NAME_AG_NEWS: "AG News",
}

model_cache: Dict[str, torch.nn.Module] = {}
nlp_cache: Dict[Tuple[str, str, int, int, int, int, int], NLPInferenceEngine] = {}


# -------------------------- Helper Functions --------------------------
def get_available_cv_models() -> List[str]:
    """Return CV models that have weights on disk."""
    available_models = []
    for model_name, weight_filename in MODEL_FILE_MAP.items():
        weight_path = CV_RESULTS_DIR / weight_filename
        if weight_path.exists():
            available_models.append(model_name)
    return available_models


def get_available_nlp_models() -> List[str]:
    """Return NLP models that have weights on disk."""
    available_models = []
    for model_name, weight_filename in NLP_MODEL_FILE_MAP.items():
        weight_path = NLP_RESULTS_DIR / weight_filename
        if weight_path.exists():
            available_models.append(model_name)
    return available_models


def get_model(model_name: str) -> torch.nn.Module:
    """Load a CV model from weights and cache it."""
    if model_name in model_cache:
        return model_cache[model_name]

    if model_name not in MODEL_FILE_MAP:
        raise ValueError(f"Unsupported model '{model_name}'.")

    weight_path = CV_RESULTS_DIR / MODEL_FILE_MAP[model_name]
    if not weight_path.exists():
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

    logger.info("Loading CV model %s from %s", model_name, weight_path)
    num_classes = 2 if DEFAULT_DATASET == dogs_vs_cats.DATASET_NAME_DOGS_VS_CATS else 10
    model = load_model(
        model_name=model_name,
        weight_path=weight_path,
        num_classes=num_classes,
        device=DEVICE,
    )
    model_cache[model_name] = model
    return model


def get_transforms(dataset_name: str):
    """Return the transform pipeline based on the dataset."""
    from torchvision import transforms

    if dataset_name == fashion_mnist.DATASET_NAME_FASHION_MNIST:
        return transforms.Compose(
            [
                transforms.Resize((DEFAULT_CROP, DEFAULT_CROP)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((DEFAULT_RESIZE, DEFAULT_RESIZE)),
            transforms.CenterCrop(DEFAULT_CROP),
            transforms.ToTensor(),
            transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )


def get_nlp_model(model_type: str, requested_dataset: str = DEFAULT_TEXT_DATASET) -> Tuple[NLPInferenceEngine, Dict[str, Any]]:
    """Load an NLP model using saved metadata when available."""
    if model_type not in NLP_MODEL_FILE_MAP:
        raise ValueError(f"Unsupported NLP model '{model_type}'.")
    if requested_dataset not in TEXT_DATASETS:
        raise ValueError(f"Unsupported dataset '{requested_dataset}'.")

    model_path = NLP_RESULTS_DIR / NLP_MODEL_FILE_MAP[model_type]
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model_config = load_model_config(model_path)
    resolved_dataset = str(model_config.get("dataset", requested_dataset))
    if resolved_dataset not in TEXT_DATASETS:
        raise ValueError(f"Unsupported dataset '{resolved_dataset}' in model config.")

    embedding_dim = int(model_config.get("embedding_dim", 128))
    hidden_dim = int(model_config.get("hidden_dim", 256))
    num_heads = int(model_config.get("num_heads", 4))
    num_layers = int(model_config.get("num_layers", 3))
    max_seq_len = int(model_config.get("max_seq_len", 256))
    num_classes = int(model_config.get("num_classes", 2 if resolved_dataset == text_data.DATASET_NAME_IMDB else 4))

    cache_key = (
        model_type,
        resolved_dataset,
        embedding_dim,
        hidden_dim,
        num_heads,
        num_layers,
        max_seq_len,
    )
    if cache_key in nlp_cache:
        return nlp_cache[cache_key], {
            "dataset": resolved_dataset,
            "display_name": TEXT_DATASETS[resolved_dataset],
            "max_seq_len": max_seq_len,
            "configured_dataset": resolved_dataset != requested_dataset,
        }

    from nlp_sources.data_processor.base import Vocabulary
    from nlp_sources.data_processor import text_data as text_dataset_loader

    vocab_filename = model_config.get("vocab_filename")
    vocab_candidates = []
    if vocab_filename:
        vocab_candidates.append(NLP_RESULTS_DIR / str(vocab_filename))
    vocab_candidates.append(NLP_RESULTS_DIR / f"vocab_{resolved_dataset}.json")

    vocab = None
    for vocab_path in vocab_candidates:
        if vocab_path.exists():
            vocab = Vocabulary.load(str(vocab_path))
            break

    if vocab is None:
        logger.info("Vocabulary file not found for %s, rebuilding from dataset %s", model_type, resolved_dataset)
        _, _, vocab, _ = text_dataset_loader.load_data(
            resolved_dataset,
            batch_size=32,
            max_seq_len=max_seq_len,
        )

    engine = NLPInferenceEngine(
        model_type=model_type,
        model_path=model_path,
        vocab=vocab,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=False,
    )
    nlp_cache[cache_key] = engine
    logger.info("NLP model %s loaded and cached", model_type)

    return engine, {
        "dataset": resolved_dataset,
        "display_name": TEXT_DATASETS[resolved_dataset],
        "max_seq_len": max_seq_len,
        "configured_dataset": resolved_dataset != requested_dataset,
    }


def nlp_predict(engine: NLPInferenceEngine, text: str, max_seq_len: int) -> tuple[int, float]:
    """Predict a class for text using NLPInferenceEngine."""
    preds, confidences = engine.predict_batch([text], max_seq_len)
    return preds[0], confidences[0][preds[0]]


def json_error(message: str, status_code: int) -> JSONResponse:
    """Return a consistent JSON error payload."""
    return JSONResponse(status_code=status_code, content={"error": message})


# -------------------------- Endpoints --------------------------
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serve the landing page."""
    available_cv_models = get_available_cv_models()
    available_nlp_models = get_available_nlp_models()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cv_model_count": len(available_cv_models),
            "nlp_model_count": len(available_nlp_models),
            "cv_ready": bool(available_cv_models),
            "nlp_ready": bool(available_nlp_models),
        },
    )


@app.get("/health")
async def health_check():
    """Expose lightweight app health information."""
    return {
        "status": "ok",
        "device": str(DEVICE),
        "cv_models": len(get_available_cv_models()),
        "nlp_models": len(get_available_nlp_models()),
    }


@app.get("/cv", response_class=HTMLResponse)
async def cv_page(request: Request):
    """Serve the CV classification page."""
    available_models = get_available_cv_models()
    default_model = DEFAULT_MODEL if DEFAULT_MODEL in available_models else (available_models[0] if available_models else None)
    return templates.TemplateResponse(
        request=request,
        name="cv.html",
        context={
            "model_name": default_model,
            "available_models": available_models,
            "has_models": bool(available_models),
        },
    )


@app.post("/cv/predict")
async def cv_predict_endpoint(file: UploadFile = File(...), model_name: str = Form(...)):
    """Handle image upload and run inference using the selected model."""
    if model_name not in get_available_cv_models():
        return json_error(f"Model '{model_name}' is not available on this server.", 404)

    if not file.filename:
        return json_error("No image file was provided.", 400)

    try:
        model = get_model(model_name)
        image_data = await file.read()
        if not image_data:
            return json_error("The uploaded image is empty.", 400)

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        preprocess = get_transforms(DEFAULT_DATASET)
        img_tensor = preprocess(img).unsqueeze(0).to(DEVICE)

        top_probs, top_indices = run_inference(model, img_tensor, top_k=2)
        dataset_labels = LABELS.get(DEFAULT_DATASET, ["Class 0", "Class 1"])
        results = []
        for prob, idx in zip(top_probs, top_indices):
            label = dataset_labels[idx] if idx < len(dataset_labels) else f"Class {idx}"
            results.append({"label": label, "confidence": round(prob * 100, 2)})

        return {
            "predictions": results,
            "top_label": results[0]["label"],
            "top_confidence": results[0]["confidence"],
            "model_name": model_name,
        }
    except UnidentifiedImageError:
        return json_error("The uploaded file is not a valid image.", 400)
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("CV inference request failed: %s", exc)
        return json_error(str(exc), 400)
    except Exception as exc:
        logger.exception("Unexpected CV inference failure")
        return json_error(f"Image classification failed: {exc}", 500)


@app.get("/nlp", response_class=HTMLResponse)
async def nlp_page(request: Request):
    """Serve the NLP classification page."""
    available_models = get_available_nlp_models()
    return templates.TemplateResponse(
        request=request,
        name="nlp.html",
        context={
            "available_models": available_models,
            "available_datasets": TEXT_DATASETS,
            "default_dataset": DEFAULT_TEXT_DATASET,
            "has_models": bool(available_models),
        },
    )


@app.post("/nlp/predict")
async def nlp_predict_endpoint(
    text: str = Form(...),
    model_type: str = Form("lstm"),
    dataset: str = Form(DEFAULT_TEXT_DATASET),
):
    """Handle text classification requests."""
    if not text or not text.strip():
        return json_error("No text provided.", 400)

    if model_type not in get_available_nlp_models():
        return json_error(f"Model '{model_type}' is not available on this server.", 404)

    try:
        engine, model_details = get_nlp_model(model_type, dataset)
        pred, confidence = nlp_predict(engine, text, model_details["max_seq_len"])
        class_names = get_class_names(model_details["dataset"])

        return {
            "prediction": class_names[pred],
            "class_id": pred,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_raw": confidence,
            "dataset": model_details["dataset"],
            "dataset_label": model_details["display_name"],
            "model_type": model_type,
            "configured_dataset": model_details["configured_dataset"],
        }
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("NLP inference request failed: %s", exc)
        return json_error(str(exc), 400)
    except Exception as exc:
        logger.exception("Unexpected NLP inference failure")
        return json_error(f"Text classification failed: {exc}", 500)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
