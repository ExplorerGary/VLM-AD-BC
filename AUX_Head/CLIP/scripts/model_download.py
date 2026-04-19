"""
Download sentence-transformers/clip-ViT-B-32 model to local model_weights directory.
"""

import argparse
import os
import sys
import ssl
import urllib.request
from pathlib import Path

# Add parent directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
CLIP_DIR = SCRIPT_DIR.parent
AUX_HEAD_DIR = CLIP_DIR.parent
PROJECT_ROOT = AUX_HEAD_DIR.parent.parent.parent  # Go up to VLM root

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description="Download CLIP-ViT-B-32 model from HuggingFace.")
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/clip-ViT-B-32",
        help="Model identifier from HuggingFace (default: sentence-transformers/clip-ViT-B-32)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(CLIP_DIR / "model_weights"),
        help="Directory to cache the model weights",
    )
    return parser.parse_args()


def fix_ssl_context():
    """Fix SSL certificate issues for HuggingFace downloads on Windows."""
    try:
        # Try to clear problematic SSL_CERT_FILE environment variable
        if "SSL_CERT_FILE" in os.environ:
            cert_file = os.environ["SSL_CERT_FILE"]
            if not os.path.exists(cert_file):
                print(f"Warning: SSL_CERT_FILE={cert_file} does not exist, removing from environment.")
                del os.environ["SSL_CERT_FILE"]
        
        # Unverified HTTPS context is needed for some environments
        ssl._create_default_https_context = ssl._create_unverified_context
        print("✓ SSL context configured")
    except Exception as e:
        print(f"Warning: Could not configure SSL: {e}")


def download_model(model_name: str, cache_dir: str):
    """Download model from HuggingFace and cache it locally."""
    fix_ssl_context()
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"Downloading model: {model_name}")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Set HuggingFace cache directory
        os.environ["HF_HOME"] = cache_dir
        os.environ["TRANSFORMERS_CACHE"] = cache_dir
        
        # Import after SSL fix
        from sentence_transformers import SentenceTransformer
        
        # Download and cache the model
        model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir,
            trust_remote_code=True,
        )
        
        print(f"✓ Model downloaded and cached successfully to: {cache_dir}")
        print(f"Model info:")
        print(f"  - Name: {model_name}")
        print(f"  - Type: {type(model)}")
        
        return model
    
    except Exception as e:
        print(f"✗ Error downloading model: {e}")
        print("\nTrying fallback: Using transformers library directly...")
        
        try:
            from transformers import AutoModel
            
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            
            print(f"✓ Model downloaded (fallback) to: {cache_dir}")
            return model
        
        except Exception as e2:
            print(f"✗ Fallback also failed: {e2}")
            raise


def main():
    args = parse_args()
    download_model(args.model_name, args.cache_dir)


if __name__ == "__main__":
    main()
