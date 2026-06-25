#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NLP Inference Framework
# -----------------------------------------------------------------------------

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from argparse import ArgumentParser

from nlp_sources.data_processor import text_data
from nlp_sources.models import lstm, gru, transformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="NLP Text Classification Inference")

    parser.add_argument("--model", type=str, required=True,
                        choices=[lstm.MODEL_TYPE_LSTM, gru.MODEL_TYPE_GRU,
                                 transformer.MODEL_TYPE_TRANSFORMER],
                        help="Select model type")
    parser.add_argument("--model-path", type=Path, required=True,
                        help="Path to trained model weights")
    parser.add_argument("--text", type=str,
                        help="Text to classify")
    parser.add_argument("--text-file", type=Path,
                        help="File containing text to classify (one per line)")
    parser.add_argument("--dataset", type=str,
                        default=text_data.DATASET_NAME_IMDB,
                        choices=[text_data.DATASET_NAME_IMDB, text_data.DATASET_NAME_AG_NEWS],
                        help="Dataset name (used to rebuild vocabulary)")
    parser.add_argument("--embedding-dim", type=int, default=128,
                        help="Embedding dimension (must match training)")
    parser.add_argument("--hidden-dim", type=int, default=256,
                        help="Hidden dimension (must match training)")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Transformer attention heads (must match training)")
    parser.add_argument("--num-layers", type=int, default=3,
                        help="Transformer encoder layers (must match training)")
    parser.add_argument("--max-seq-len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--no-cuda", action="store_true",
                        help="Force use CPU")
    parser.add_argument("--fp16", action="store_true",
                        help="Use half-precision inference")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference")
    parser.add_argument("--output-json", action="store_true",
                        help="Output results in JSON format")

    return parser.parse_args()


def load_model_config(model_path: Path) -> dict:
    """Load optional model metadata saved alongside the weights."""
    config_candidates = [model_path.with_suffix(".json"), model_path.parent / f"{model_path.stem}.json"]
    for config_path in config_candidates:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
    return {}


def load_model_state_dict(model_path: Path, device: str | torch.device = "cpu") -> Dict[str, torch.Tensor]:
    """Load model weights from disk."""
    return torch.load(model_path, map_location=device, weights_only=True)


def _set_resolved_config_value(
    resolved_config: Dict[str, Any],
    field_name: str,
    inferred_value: Any,
) -> None:
    current_value = resolved_config.get(field_name)
    if current_value is not None and current_value != inferred_value:
        logger.warning(
            "Overriding saved model config %s=%s with checkpoint-derived value %s",
            field_name,
            current_value,
            inferred_value,
        )
    resolved_config[field_name] = inferred_value


def _infer_recurrent_model_config(
    state_dict: Dict[str, torch.Tensor],
    module_name: str,
) -> Dict[str, Any]:
    layer_indices = set()
    bidirectional = False

    for key in state_dict:
        prefix = f"{module_name}.weight_ih_l"
        if not key.startswith(prefix):
            continue

        suffix = key[len(prefix):]
        layer_index_text = suffix.split("_", 1)[0]
        layer_indices.add(int(layer_index_text))
        bidirectional = bidirectional or suffix.endswith("_reverse")

    hidden_weight_key = f"{module_name}.weight_hh_l0"
    return {
        "embedding_dim": int(state_dict["embedding.weight"].shape[1]),
        "hidden_dim": int(state_dict[hidden_weight_key].shape[1]),
        "num_classes": int(state_dict["fc.weight"].shape[0]),
        "num_layers": len(layer_indices),
        "bidirectional": bidirectional,
    }


def _infer_transformer_model_config(state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    layer_indices = set()
    hidden_dim = None

    for key in state_dict:
        layer_match = re.search(r"(?:^|\.)(?:transformer_encoder|encoder)\.layers\.(\d+)\.", key)
        if layer_match:
            layer_indices.add(int(layer_match.group(1)))

        if hidden_dim is None and key.endswith(".linear1.weight"):
            hidden_dim = int(state_dict[key].shape[0])

    if hidden_dim is None:
        raise KeyError("Unable to infer transformer hidden_dim from checkpoint state_dict")

    return {
        "embedding_dim": int(state_dict["embedding.weight"].shape[1]),
        "hidden_dim": hidden_dim,
        "num_classes": int(state_dict["fc.weight"].shape[0]),
        "num_layers": len(layer_indices),
        "max_seq_len": int(state_dict["pos_encoder.pe"].shape[1]),
    }


def resolve_model_runtime_config(
    model_type: str,
    model_path: Path,
    model_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve model hyperparameters from config, correcting stale values from the checkpoint when needed."""
    resolved_config: Dict[str, Any] = dict(model_config or {})
    state_dict = load_model_state_dict(model_path, device="cpu")

    if model_type == lstm.MODEL_TYPE_LSTM:
        inferred_config = _infer_recurrent_model_config(state_dict, "lstm")
    elif model_type == gru.MODEL_TYPE_GRU:
        inferred_config = _infer_recurrent_model_config(state_dict, "gru")
    elif model_type == transformer.MODEL_TYPE_TRANSFORMER:
        inferred_config = _infer_transformer_model_config(state_dict)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    for field_name, inferred_value in inferred_config.items():
        _set_resolved_config_value(resolved_config, field_name, inferred_value)

    return resolved_config


def load_vocab(dataset: str, model_path: Path, vocab_filename: Optional[str] = None):
    """Load a saved vocabulary if available; otherwise rebuild it from the dataset."""
    candidate_paths = []
    if vocab_filename:
        candidate_paths.append(model_path.parent / vocab_filename)
    candidate_paths.append(model_path.parent / f"vocab_{dataset}.json")

    for vocab_path in candidate_paths:
        if vocab_path.exists():
            logger.info(f"Loading vocabulary from {vocab_path}...")
            from nlp_sources.data_processor.base import Vocabulary
            return Vocabulary.load(str(vocab_path))

    logger.info(f"Rebuilding vocabulary from {dataset}...")
    _, _, vocab, _ = text_data.load_data(dataset, batch_size=1, max_seq_len=512)
    return vocab


class NLPInferenceEngine:
    """
    NLP Inference Engine for text classification.
    
    Provides batch inference, model optimization, and structured output.
    """
    
    def __init__(self, model_type: str, model_path: Path, vocab, num_classes: int,
                 embedding_dim: int = 128, hidden_dim: int = 256, 
                 device: str = "cpu", fp16: bool = False,
                 num_heads: int = 4, num_layers: int = 3, max_seq_len: int = 512,
                 bidirectional: bool = True):
        self.model_type = model_type
        self.model_path = model_path
        self.vocab = vocab
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.fp16 = fp16
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.bidirectional = bidirectional
        
        self.model = self._build_model(embedding_dim, hidden_dim)
        self._load_weights()
        
    def _build_model(self, embedding_dim: int, hidden_dim: int):
        """Build the appropriate model based on type."""
        logger.info(f"Building {self.model_type} model...")
        
        model_map = {
            lstm.MODEL_TYPE_LSTM: lstm.lstm_classifier,
            gru.MODEL_TYPE_GRU: gru.gru_classifier,
            transformer.MODEL_TYPE_TRANSFORMER: transformer.transformer_classifier
        }
        
        model_fn = model_map.get(self.model_type)
        if not model_fn:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        return model_fn(
            vocab_size=self.vocab.size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_classes=self.num_classes,
            bidirectional=self.bidirectional,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            max_seq_len=self.max_seq_len,
            padding_idx=0,
        ).to(self.device)
    
    def _load_weights(self):
        """Load trained model weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        logger.info(f"Loading model weights from {self.model_path}...")
        state_dict = load_model_state_dict(self.model_path, device=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        if self.fp16 and self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Enabled half-precision inference")
        elif self.fp16:
            logger.warning("fp16 requested on CPU; using float32 inference instead")
    
    def preprocess_texts(self, texts: List[str], max_seq_len: int) -> torch.Tensor:
        """Preprocess a list of texts into model input tensors."""
        indexed_texts = []
        for text in texts:
            tokens = self.vocab.tokenize(text)
            indexed = self.vocab.encode(tokens, max_seq_len)
            indexed_texts.append(indexed)
        
        return torch.tensor(indexed_texts, dtype=torch.long).to(self.device)
    
    @torch.no_grad()
    def predict_batch(self, texts: List[str], max_seq_len: int = 512) -> Tuple[List[int], List[List[float]]]:
        """
        Predict classes for a batch of texts.
        
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        if not texts:
            return [], []
        
        input_tensor = self.preprocess_texts(texts, max_seq_len)
        
        outputs = self.model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1).tolist()
        confidences = probs.tolist()
        
        return preds, confidences


def get_class_names(dataset: str) -> List[str]:
    """Get class names based on dataset type."""
    if dataset == text_data.DATASET_NAME_IMDB:
        return ["Negative", "Positive"]
    else:
        return ["World", "Sports", "Business", "Sci/Tech"]


def main():
    args = parse_args()
    
    try:
        device = "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
        logger.info(f"Using device: {device}")

        model_config = resolve_model_runtime_config(args.model, args.model_path, load_model_config(args.model_path))
        dataset = model_config.get("dataset", args.dataset)
        vocab = load_vocab(dataset, args.model_path, model_config.get("vocab_filename"))

        if model_config:
            logger.info(f"Loaded model config from disk: {model_config}")

        if "num_classes" in model_config:
            num_classes = int(model_config["num_classes"])
        else:
            _, _, _, num_classes = text_data.load_data(
                dataset,
                batch_size=args.batch_size,
                max_seq_len=args.max_seq_len
            )

        logger.info(f"Vocabulary size: {vocab.size}")
        
        engine = NLPInferenceEngine(
            model_type=args.model,
            model_path=args.model_path,
            vocab=vocab,
            num_classes=num_classes,
            embedding_dim=int(model_config.get("embedding_dim", args.embedding_dim)),
            hidden_dim=int(model_config.get("hidden_dim", args.hidden_dim)),
            device=device,
            fp16=args.fp16,
            num_heads=int(model_config.get("num_heads", args.num_heads)),
            num_layers=int(model_config.get("num_layers", args.num_layers)),
            max_seq_len=int(model_config.get("max_seq_len", args.max_seq_len)),
            bidirectional=bool(model_config.get("bidirectional", True)),
        )
        
        texts = []
        if args.text:
            texts = [args.text]
        elif args.text_file and args.text_file.exists():
            with open(args.text_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            logger.error("Please provide --text or --text-file")
            sys.exit(1)
        
        class_names = get_class_names(dataset)
        results = []
        
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i+args.batch_size]
            preds, confidences = engine.predict_batch(batch, args.max_seq_len)
            
            for text, pred, conf in zip(batch, preds, confidences):
                results.append({
                    "text": text,
                    "prediction": class_names[pred],
                    "class_index": pred,
                    "confidence": conf[pred],
                    "all_confidences": dict(zip(class_names, conf))
                })
        
        if args.output_json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            print(f"\n{'='*60}")
            for result in results:
                display_text = result["text"][:60] + "..." if len(result["text"]) > 60 else result["text"]
                print(f"Text: {display_text}")
                print(f"Prediction: {result['prediction']} (class {result['class_index']})")
                print(f"Confidence: {result['confidence']:.4f}")
                print(f"{'-'*60}")
        
        logger.info(f"Processed {len(texts)} texts successfully")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
