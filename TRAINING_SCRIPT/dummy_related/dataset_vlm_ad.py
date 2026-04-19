"""Dataset loader for VLM-AD training with CLIP embeddings and action labels."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from PIL import Image


class VLMADDataset(Dataset):
    """
    Dataset for VLM-AD training with CLIP embeddings and action labels.
    
    Loads data from HuggingFace dataset format with:
    - image: front camera view
    - steering, throttle, brake: regression targets
    - clip_current_action/next_action/reasoning: CLIP text embeddings (512-dim)
    - one_hot_control_flag/turn_flag/lane_flag: action classification labels
    """
    
    def __init__(self, dataset_path, split="train", max_samples=None):
        """
        Args:
            dataset_path: path to HF dataset root (containing train/ subdirectory)
            split: "train" (only train available in MERGED_RESULT)
            max_samples: max number of samples to load (for debugging)
        """
        # Load HF dataset from local directory
        from datasets import load_from_disk
        
        try:
            # First try loading as combined dataset
            self.hf_dataset = load_from_disk(dataset_path)
            if isinstance(self.hf_dataset, dict):
                self.hf_dataset = self.hf_dataset[split]
        except Exception as e:
            print(f"Failed to load from {dataset_path}: {e}")
            print("Attempting to load from specific split directory...")
            # Try loading specific split
            split_path = os.path.join(dataset_path, split)
            self.hf_dataset = load_from_disk(split_path)
        
        if max_samples and len(self.hf_dataset) > max_samples:
            indices = np.random.choice(len(self.hf_dataset), max_samples, replace=False)
            self.hf_dataset = self.hf_dataset.select(indices)
        
        self.split = split
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        
        # Load and preprocess image
        image = sample["image"]
        if isinstance(image, Image.Image):
            image_array = np.array(image, dtype=np.float32) / 255.0
        else:
            image_array = np.array(image, dtype=np.float32) / 255.0
        
        # Flatten to [4200] for compatibility with Dummy model
        image_tensor = torch.from_numpy(image_array).flatten()
        
        # Regression targets
        steering = torch.tensor(sample["steering"], dtype=torch.float32)
        throttle = torch.tensor(sample["throttle"], dtype=torch.float32)
        brake = torch.tensor(sample["brake"], dtype=torch.float32)
        regression_targets = torch.stack([steering, throttle, brake])
        
        # CLIP embeddings (512-dim)
        clip_current = torch.tensor(sample["clip_current_action"], dtype=torch.float32)
        clip_next = torch.tensor(sample["clip_next_action"], dtype=torch.float32)
        clip_reasoning = torch.tensor(sample["clip_reasoning"], dtype=torch.float32)
        
        # One-hot flags to indices (for CrossEntropy)
        # One-hot format: list of 0s and 1s
        control_label = torch.argmax(torch.tensor(sample["one_hot_control_flag"], dtype=torch.int64))
        turn_label = torch.argmax(torch.tensor(sample["one_hot_turn_flag"], dtype=torch.int64))
        lane_label = torch.argmax(torch.tensor(sample["one_hot_lane_flag"], dtype=torch.int64))
        
        return {
            "image": image_tensor,
            "steering": steering,
            "throttle": throttle,
            "brake": brake,
            "regression_targets": regression_targets,
            "clip_current": clip_current,
            "clip_next": clip_next,
            "clip_reasoning": clip_reasoning,
            "control_label": control_label,
            "turn_label": turn_label,
            "lane_label": lane_label,
        }


def collate_fn_vlm_ad(batch):
    """Collate function for VLM-AD dataset."""
    images = torch.stack([b["image"] for b in batch])
    regression_targets = torch.stack([b["regression_targets"] for b in batch])
    
    # CLIP embeddings
    clip_currents = torch.stack([b["clip_current"] for b in batch])
    clip_nexts = torch.stack([b["clip_next"] for b in batch])
    clip_reasonings = torch.stack([b["clip_reasoning"] for b in batch])
    
    # Action labels
    control_labels = torch.stack([b["control_label"] for b in batch])
    turn_labels = torch.stack([b["turn_label"] for b in batch])
    lane_labels = torch.stack([b["lane_label"] for b in batch])
    
    return {
        "images": images,
        "regression_targets": regression_targets,
        "clip_current": clip_currents,
        "clip_next": clip_nexts,
        "clip_reasoning": clip_reasonings,
        "control_labels": control_labels,
        "turn_labels": turn_labels,
        "lane_labels": lane_labels,
    }
