#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# NLP Inference Framework
# -----------------------------------------------------------------------------

import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

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
    parser.add_argument("--max-seq-len", type=int, default=256,
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


class NLPInferenceEngine:
    """
    NLP Inference Engine for text classification.
    
    Provides batch inference, model optimization, and structured output.
    """
    
    def __init__(self, model_type: str, model_path: Path, vocab, num_classes: int,
                 embedding_dim: int = 128, hidden_dim: int = 256, 
                 device: str = "cpu", fp16: bool = False):
        self.model_type = model_type
        self.model_path = model_path
        self.vocab = vocab
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.fp16 = fp16
        
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
            num_classes=self.num_classes
        ).to(self.device)
    
    def _load_weights(self):
        """Load trained model weights."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model weights not found: {self.model_path}")
        
        logger.info(f"Loading model weights from {self.model_path}...")
        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        if self.fp16:
            self.model = self.model.half()
            logger.info("Enabled half-precision inference")
    
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
        
        if self.fp16:
            input_tensor = input_tensor.half()
        
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
        
        logger.info(f"Rebuilding vocabulary from {args.dataset}...")
        _, _, vocab, num_classes = text_data.load_data(
            args.dataset, 
            batch_size=args.batch_size, 
            max_seq_len=args.max_seq_len
        )
        logger.info(f"Vocabulary size: {vocab.size}")
        
        engine = NLPInferenceEngine(
            model_type=args.model,
            model_path=args.model_path,
            vocab=vocab,
            num_classes=num_classes,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            device=device,
            fp16=args.fp16
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
        
        class_names = get_class_names(args.dataset)
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
