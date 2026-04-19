import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

# PROJECT ROOT
SCRIPT_DIR = Path(__file__).resolve().parent
CLIP_DIR = SCRIPT_DIR.parent
AUX_HEAD_DIR = CLIP_DIR.parent
PROJECT_ROOT = AUX_HEAD_DIR.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Global model instance
_model = None


def load_model(model_path=None):
    """
    Load CLIP model from local cache or specified path.
    
    Args:
        model_path: Optional path to model. If None, uses default local cache.
    
    Returns:
        SentenceTransformer model instance
    """
    global _model
    
    if model_path is None:
        model_dir = CLIP_DIR / "model_weights" / "models--sentence-transformers--clip-ViT-B-32" / "snapshots"
        
        if model_dir.exists():
            snapshots = list(model_dir.glob("*"))
            if snapshots:
                model_path = str(snapshots[0])
            else:
                raise FileNotFoundError(f"No model snapshots found in {model_dir}")
        else:
            raise FileNotFoundError(
                f"Model directory not found: {model_dir}\n"
                f"Please run model_download.py first."
            )
    
    print(f"Loading CLIP model from: {model_path}")
    _model = SentenceTransformer(model_path)
    print(f"✓ Model loaded: {type(_model)}")
    return _model


def get_model():
    """Get global model instance. Load if not already loaded."""
    global _model
    if _model is None:
        load_model()
    return _model


def encode_images(image_paths, batch_size=32, show_progress=True):
    """
    Encode a list of images to embeddings.
    
    Args:
        image_paths: List of image file paths
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        numpy array of shape (num_images, embedding_dim)
    """
    model = get_model()
    images = []
    
    iterator = tqdm(image_paths, desc="Loading images", disable=not show_progress)
    for img_path in iterator:
        try:
            img = Image.open(img_path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {img_path}: {e}")
    
    print(f"Encoding {len(images)} images...")
    embeddings = model.encode(images, batch_size=batch_size, show_progress_bar=show_progress)
    return embeddings


def encode_texts(texts, batch_size=32, show_progress=True):
    """
    Encode a list of texts to embeddings.
    
    Args:
        texts: List of text strings
        batch_size: Batch size for encoding
        show_progress: Show progress bar
    
    Returns:
        numpy array of shape (num_texts, embedding_dim)
    """
    model = get_model()
    print(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress)
    return embeddings


def compute_similarities(image_embeddings, text_embeddings):
    """
    Compute cosine similarities between image and text embeddings.
    
    Args:
        image_embeddings: numpy array of shape (num_images, embedding_dim)
        text_embeddings: numpy array of shape (num_texts, embedding_dim)
    
    Returns:
        numpy array of shape (num_images, num_texts) with similarity scores
    """
    cos_scores = util.cos_sim(image_embeddings, text_embeddings)
    return cos_scores.cpu().numpy() if isinstance(cos_scores, torch.Tensor) else cos_scores


def dummy_eval():
    """Dummy evaluation with hardcoded images and texts."""
    print("\n=== Running Dummy CLIP Evaluation ===\n")
    
    model = get_model()
    
    # Dummy image (create a simple RGB image)
    dummy_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    print("Created dummy image (224x224)")
    
    # Encode image
    img_emb = model.encode([dummy_img])
    
    # Encode text descriptions
    texts = [
        "Two dogs in the snow",
        "A cat on a table",
        "A picture of London at night",
    ]
    text_emb = model.encode(texts)
    
    # Compute cosine similarities
    cos_scores = compute_similarities(img_emb, text_emb)
    
    print("\nSimilarity scores:")
    for i, text in enumerate(texts):
        print(f"  '{text}': {cos_scores[0, i]:.4f}")
    
    return cos_scores


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP model evaluation on image-text pairs.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to CLIP model. If None, uses local cache.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="dummy",
        choices=["dummy", "dataset"],
        help="Evaluation mode: 'dummy' for hardcoded test, 'dataset' for real data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model globally
    load_model(args.model_path)
    
    if args.mode == "dummy":
        results = dummy_eval()
    else:
        raise NotImplementedError(f"Mode '{args.mode}' not yet implemented.")
    
    # Save results if output path provided
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output_data = {
            "mode": args.mode,
            "model_path": str(args.model_path or "local_cache"),
            "batch_size": args.batch_size,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"✓ Results saved to: {args.output}")


if __name__ == "__main__":
    main()