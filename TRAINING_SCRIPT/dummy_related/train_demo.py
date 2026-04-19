"""Behavior cloning training entry.

支持:
1. 可调 load()，按 driving_tasks 和目标项筛选训练数据（默认全开）
2. Trainer 训练循环与验证
3. 两个模型二选一（nvidia / nvidia_parallel）
4. 模型按 "模型名 + 时间戳" 保存到 out 目录（.pth）
"""

import argparse
import ast
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
from torch.utils.data import DataLoader, Dataset

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from AutoDriveModels.Dummy.model import NetworkNvidia, NetworkNvidiaParallel


DRIVING_TASKS = [
	"S1_normal",
	"S2_noWarning",
	"S3_rainy",
	"S4_curve",
	"S5_nighttime",
	"S6_truck",
]

DEFAULT_TARGET_ITEMS = ["Steering", "Throttle", "Brake"]


def seed_everything(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def parse_list_arg(value: Optional[str]) -> Optional[List[str]]:
	if value is None:
		return None
	parts = [part.strip() for part in value.split(",")]
	parsed = [part for part in parts if part]
	return parsed if parsed else None


def load_metadata(data_dir: str) -> Dict[str, object]:
	metadata_path = os.path.join(data_dir, "metadata.txt")
	if not os.path.exists(metadata_path):
		raise FileNotFoundError(f"metadata.txt not found: {metadata_path}")

	metadata: Dict[str, object] = {}
	with open(metadata_path, "r", encoding="utf-8") as file:
		for line in file:
			text = line.strip()
			if not text or ":" not in text:
				continue
			key, raw_value = text.split(":", 1)
			key = key.strip()
			raw_value = raw_value.strip()

			try:
				value = ast.literal_eval(raw_value)
			except (ValueError, SyntaxError):
				value = raw_value
			metadata[key] = value
	return metadata


class NpzDrivingDataset(Dataset):
	"""读取 front_camera/*.npz 并按场景和目标列筛选。"""

	def __init__(
		self,
		npz_path: str,
		feature_names: Sequence[str],
		target_items: Sequence[str],
		include_tasks: Sequence[str],
		max_samples: int = 0,
		seed: int = 42,
	) -> None:
		if not os.path.exists(npz_path):
			raise FileNotFoundError(f"Dataset file not found: {npz_path}")

		data = np.load(npz_path, allow_pickle=True)
		images = np.asarray(data["images"], dtype=np.float32)
		vehicle_data = np.asarray(data["vehicle_data"], dtype=np.float32)
		scene_names = np.asarray(data["scene_names"]).astype(str)
		timestamps = np.asarray(data["timestamps"]).astype(str)

		scene_mask = np.isin(scene_names, np.asarray(include_tasks, dtype=str))
		images = images[scene_mask]
		vehicle_data = vehicle_data[scene_mask]
		scene_names = scene_names[scene_mask]
		timestamps = timestamps[scene_mask]

		if max_samples > 0 and len(images) > max_samples:
			rng = np.random.default_rng(seed)
			picked = rng.choice(len(images), size=max_samples, replace=False)
			images = images[picked]
			vehicle_data = vehicle_data[picked]
			scene_names = scene_names[picked]
			timestamps = timestamps[picked]

		target_indices = [feature_names.index(item) for item in target_items]
		targets = vehicle_data[:, target_indices]

		self.images = torch.from_numpy(images)
		self.targets = torch.from_numpy(targets)
		self.scene_names = scene_names.tolist()
		self.timestamps = timestamps.tolist()
		self.target_items = list(target_items)

	def __len__(self) -> int:
		return int(self.images.shape[0])

	def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
		return self.images[index], self.targets[index]


def load(
	data_dir: Optional[str] = None,
	target_items: Optional[Sequence[str]] = None,
	include_tasks: Optional[Sequence[str]] = None,
	batch_size: int = 64,
	num_workers: int = 0,
	pin_memory: bool = True,
	max_train_samples: int = 0,
	max_val_samples: int = 0,
	seed: int = 42,
) -> Tuple[NpzDrivingDataset, NpzDrivingDataset, DataLoader, DataLoader, Dict[str, object]]:
	"""加载训练/验证集。

	可调超参数:
	- target_items: 训练哪些控制量（默认 Steering/Throttle/Brake 全开）
	- include_tasks: 使用哪些 driving tasks（默认 S1-S6 全开）
	- max_train_samples/max_val_samples: 可选下采样，用于快速试验
	"""
	if data_dir is None:
		this_dir = os.path.dirname(__file__)
		data_dir = os.path.join(this_dir, "dataset", "front_camera")

	metadata = load_metadata(data_dir)
	feature_names = metadata.get("feature_names")
	if not isinstance(feature_names, list):
		raise ValueError("metadata.txt missing a valid feature_names list")

	chosen_targets = list(target_items) if target_items else list(DEFAULT_TARGET_ITEMS)
	chosen_tasks = list(include_tasks) if include_tasks else list(DRIVING_TASKS)

	missing_targets = [name for name in chosen_targets if name not in feature_names]
	if missing_targets:
		raise ValueError(f"Unknown target_items: {missing_targets}; available={feature_names}")

	invalid_tasks = [name for name in chosen_tasks if name not in DRIVING_TASKS]
	if invalid_tasks:
		raise ValueError(f"Unknown driving tasks: {invalid_tasks}; available={DRIVING_TASKS}")

	train_path = os.path.join(data_dir, "train.npz")
	val_path = os.path.join(data_dir, "val.npz")

	trainingset = NpzDrivingDataset(
		npz_path=train_path,
		feature_names=feature_names,
		target_items=chosen_targets,
		include_tasks=chosen_tasks,
		max_samples=max_train_samples,
		seed=seed,
	)
	validateset = NpzDrivingDataset(
		npz_path=val_path,
		feature_names=feature_names,
		target_items=chosen_targets,
		include_tasks=chosen_tasks,
		max_samples=max_val_samples,
		seed=seed + 1,
	)

	if len(trainingset) == 0:
		raise RuntimeError("Training set is empty after filtering")
	if len(validateset) == 0:
		raise RuntimeError("Validation set is empty after filtering")

	use_pin_memory = bool(pin_memory and torch.cuda.is_available())
	trainloader = DataLoader(
		trainingset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=use_pin_memory,
	)
	validationloader = DataLoader(
		validateset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=use_pin_memory,
	)

	info = {
		"data_dir": data_dir,
		"feature_names": feature_names,
		"target_items": chosen_targets,
		"include_tasks": chosen_tasks,
		"train_scene_counts": dict(Counter(trainingset.scene_names)),
		"val_scene_counts": dict(Counter(validateset.scene_names)),
	}

	return trainingset, validateset, trainloader, validationloader, info


def build_model(model_name: str, output_dim: int) -> nn.Module:
	if model_name == "nvidia":
		return NetworkNvidia(output_dim=output_dim)
	if model_name == "nvidia_parallel":
		return NetworkNvidiaParallel(output_dim=output_dim)
	raise ValueError(f"Unsupported model: {model_name}")


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float, momentum: float):
	name = name.lower()
	if name == "adam":
		return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
	if name == "adamw":
		return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
	if name == "sgd":
		return SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
	raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
	optimizer: torch.optim.Optimizer,
	name: str,
	epochs: int,
	step_size: int,
	gamma: float,
	min_lr: float,
	plateau_patience: int,
):
	name = name.lower()
	if name == "none":
		return None
	if name == "step":
		return StepLR(optimizer, step_size=max(1, step_size), gamma=gamma)
	if name == "cosine":
		return CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
	if name == "plateau":
		return ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=max(1, plateau_patience))
	raise ValueError(f"Unsupported scheduler: {name}")


def build_criterion(loss_name: str) -> nn.Module:
	loss_name = loss_name.lower()
	if loss_name == "mse":
		return nn.MSELoss()
	if loss_name == "l1":
		return nn.L1Loss()
	if loss_name == "huber":
		return nn.SmoothL1Loss()
	raise ValueError(f"Unsupported loss: {loss_name}")


def _is_current_cuda_arch_supported() -> bool:
	if not torch.cuda.is_available():
		return False

	try:
		major, minor = torch.cuda.get_device_capability(0)
		cap = f"sm_{major}{minor}"
		supported_arches = torch.cuda.get_arch_list()
		return cap in supported_arches
	except Exception:
		return False


def resolve_device(device_name: str) -> torch.device:
	if device_name == "auto":
		if torch.cuda.is_available() and _is_current_cuda_arch_supported():
			print("Congrats: Using CUDA device.")
			return torch.device("cuda")
		if torch.cuda.is_available():
			print("Warning: CUDA is visible but current GPU arch is not supported by this PyTorch build; fallback to CPU.")
		return torch.device("cpu")
	return torch.device(device_name)


class Trainer:
	"""通用 BC Trainer。"""

	def __init__(
		self,
		ckptroot: str,
		run_name: str,
		model: nn.Module,
		device: torch.device,
		epochs: int,
		criterion: nn.Module,
		optimizer: torch.optim.Optimizer,
		scheduler,
		start_epoch: int,
		trainloader: DataLoader,
		validationloader: DataLoader,
		save_every: int,
		target_items: Sequence[str],
		model_name: str,
	) -> None:
		self.model = model
		self.device = device
		self.epochs = epochs
		self.ckptroot = ckptroot
		self.run_name = run_name
		self.criterion = criterion
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.start_epoch = start_epoch
		self.trainloader = trainloader
		self.validationloader = validationloader
		self.save_every = max(1, save_every)
		self.target_items = list(target_items)
		self.model_name = model_name
		self.best_val_loss = float("inf")

	def _run_epoch(self, is_train: bool) -> float:
		loader = self.trainloader if is_train else self.validationloader
		self.model.train(mode=is_train)
		running_loss = 0.0
		total_steps = 0

		for images, targets in loader:
			images = images.to(self.device, non_blocking=True)
			targets = targets.to(self.device, non_blocking=True)

			if is_train:
				self.optimizer.zero_grad(set_to_none=True)

			with torch.set_grad_enabled(is_train):
				outputs = self.model(images)
				loss = self.criterion(outputs, targets)

				if is_train:
					loss.backward()
					self.optimizer.step()

			running_loss += float(loss.item())
			total_steps += 1

		return running_loss / max(1, total_steps)

	def _save_checkpoint(self, filename: str, epoch: int, train_loss: float, val_loss: float) -> str:
		os.makedirs(self.ckptroot, exist_ok=True)
		ckpt_path = os.path.join(self.ckptroot, filename)
		state = {
			"epoch": epoch,
			"state_dict": self.model.state_dict(),
			"optimizer": self.optimizer.state_dict(),
			"scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
			"best_val_loss": self.best_val_loss,
			"model_name": self.model_name,
			"target_items": self.target_items,
			"train_loss": train_loss,
			"val_loss": val_loss,
		}
		torch.save(state, ckpt_path)
		return ckpt_path

	def train(self) -> None:
		self.model.to(self.device)

		end_epoch = self.start_epoch + self.epochs
		for epoch in range(self.start_epoch, end_epoch):
			train_loss = self._run_epoch(is_train=True)
			val_loss = self._run_epoch(is_train=False)

			if self.scheduler is not None:
				if isinstance(self.scheduler, ReduceLROnPlateau):
					self.scheduler.step(val_loss)
				else:
					self.scheduler.step()

			lr = self.optimizer.param_groups[0]["lr"]
			print(
				f"Epoch [{epoch + 1}/{end_epoch}] "
				f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.6e}"
			)

			if val_loss < self.best_val_loss:
				self.best_val_loss = val_loss
				best_name = f"{self.run_name}_best.pth"
				best_path = self._save_checkpoint(best_name, epoch + 1, train_loss, val_loss)
				print(f"  Saved best checkpoint: {best_path}")

			is_save_epoch = ((epoch + 1) % self.save_every == 0) or (epoch + 1 == end_epoch)
			if is_save_epoch:
				epoch_name = f"{self.run_name}_epoch{epoch + 1:03d}.pth"
				epoch_path = self._save_checkpoint(epoch_name, epoch + 1, train_loss, val_loss)
				print(f"  Saved checkpoint: {epoch_path}")


def main() -> None:
	parser = argparse.ArgumentParser(description="Behavior Cloning Trainer")

	parser.add_argument("--data-dir", type=str, default=None, help="Path to dataset/front_camera")
	parser.add_argument("--model", type=str, default="nvidia_parallel", choices=["nvidia", "nvidia_parallel"])
	parser.add_argument(
		"--targets",
		type=str,
		default=",".join(DEFAULT_TARGET_ITEMS),
		help="Comma-separated target items, e.g. Steering,Throttle,Brake",
	)
	parser.add_argument(
		"--tasks",
		type=str,
		default=",".join(DRIVING_TASKS),
		help="Comma-separated driving tasks to include",
	)

	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--max-train-samples", type=int, default=0)
	parser.add_argument("--max-val-samples", type=int, default=0)

	parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"])
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--momentum", type=float, default=0.9)

	parser.add_argument("--loss", type=str, default="mse", choices=["mse", "l1", "huber"])
	parser.add_argument("--scheduler", type=str, default="cosine", choices=["none", "step", "cosine", "plateau"])
	parser.add_argument("--step-size", type=int, default=10)
	parser.add_argument("--gamma", type=float, default=0.5)
	parser.add_argument("--min-lr", type=float, default=1e-6)
	parser.add_argument("--plateau-patience", type=int, default=3)

	parser.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda")
	parser.add_argument("--save-every", type=int, default=5)
	parser.add_argument("--resume", type=str, default="", help="Checkpoint path to resume")

	args = parser.parse_args()

	seed_everything(args.seed)
	device = resolve_device(args.device)
	this_dir = os.path.dirname(__file__)
	out_dir = os.path.join(this_dir, "out")
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	run_name = f"{args.model}_{timestamp}"

	target_items = parse_list_arg(args.targets)
	include_tasks = parse_list_arg(args.tasks)

	trainingset, validateset, trainloader, validationloader, info = load(
		data_dir=args.data_dir,
		target_items=target_items,
		include_tasks=include_tasks,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=True,
		max_train_samples=args.max_train_samples,
		max_val_samples=args.max_val_samples,
		seed=args.seed,
	)

	output_dim = len(info["target_items"])
	model = build_model(args.model, output_dim=output_dim)
	criterion = build_criterion(args.loss)
	optimizer = build_optimizer(model, args.optimizer, args.lr, args.weight_decay, args.momentum)
	scheduler = build_scheduler(
		optimizer=optimizer,
		name=args.scheduler,
		epochs=args.epochs,
		step_size=args.step_size,
		gamma=args.gamma,
		min_lr=args.min_lr,
		plateau_patience=args.plateau_patience,
	)

	start_epoch = 0
	best_val_loss = float("inf")
	if args.resume:
		checkpoint = torch.load(args.resume, map_location=device)
		model.load_state_dict(checkpoint["state_dict"])
		if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
			optimizer.load_state_dict(checkpoint["optimizer"])
		if scheduler is not None and "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
			scheduler.load_state_dict(checkpoint["scheduler"])
		start_epoch = int(checkpoint.get("epoch", 0))
		best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
		print(f"Resumed from {args.resume}, start_epoch={start_epoch}, best_val_loss={best_val_loss:.6f}")

	print("=" * 80)
	print(f"Device: {device}")
	print(f"Model: {args.model}")
	print(f"Targets: {info['target_items']}")
	print(f"Driving tasks: {info['include_tasks']}")
	print(f"Train samples: {len(trainingset)} | Val samples: {len(validateset)}")
	print(f"Train scene counts: {info['train_scene_counts']}")
	print(f"Val scene counts: {info['val_scene_counts']}")
	print(f"Output dir: {out_dir}")
	print("=" * 80)

	trainer = Trainer(
		ckptroot=out_dir,
		run_name=run_name,
		model=model,
		device=device,
		epochs=args.epochs,
		criterion=criterion,
		optimizer=optimizer,
		scheduler=scheduler,
		start_epoch=start_epoch,
		trainloader=trainloader,
		validationloader=validationloader,
		save_every=args.save_every,
		target_items=info["target_items"],
		model_name=args.model,
	)
	trainer.best_val_loss = best_val_loss
	trainer.train()

	print("Training finished.")


if __name__ == "__main__":
	main()
